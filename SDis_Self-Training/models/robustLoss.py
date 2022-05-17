import numbers
import mpmath
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

def get_resource_as_file(path):
  class NullContextManager(object):
    def __init__(self, dummy_resource=None): self.dummy_resource = dummy_resource
    def __enter__(self): return self.dummy_resource
    def __exit__(self, *args): pass
  return NullContextManager('./' + path)
  
def log_safe(x):
  return tf.math.log(tf.minimum(x, tf.cast(3e37, x.dtype)))

def log1p_safe(x):
  return tf.math.log1p(tf.minimum(x, tf.cast(3e37, x.dtype)))

def exp_safe(x):
  return tf.math.exp(tf.minimum(x, tf.cast(87.5, x.dtype)))

def expm1_safe(x):
  return tf.math.expm1(tf.minimum(x, tf.cast(87.5, x.dtype)))

def inv_softplus(y):
  return tf.where(y > 87.5, y, tf.math.log(tf.math.expm1(y)))

def affine_sigmoid(real, lo=0, hi=1):
  alpha = tf.sigmoid(real) * (hi - lo) + lo
  return alpha

def inv_affine_sigmoid(alpha, lo=0, hi=1):  
  y = ((alpha - lo) / (hi - lo))
  return -tf.math.log(1. / y - 1.)

def affine_softplus(real, lo=0, ref=1):
  shift = inv_softplus(tf.cast(1., real.dtype))
  scale = (ref - lo) * tf.nn.softplus(real + shift) + lo
  return scale

def partition_spline_curve(alpha):
  c = lambda z: tf.cast(z, alpha.dtype)
  x = tf.where(alpha < 4, (c(2.25) * alpha - c(4.5)) / (tf.abs(alpha - c(2)) + c(0.25)) + alpha + c(2), c(5) / c(18) * log_safe(c(4) * alpha - c(15)) + c(8))
  return x

def lossfun(x, alpha, scale, approximate=False, epsilon=1e-6):
  float_dtype = x.dtype
  alpha = tf.broadcast_to(alpha, tf.shape(x))
  scale = tf.broadcast_to(scale, tf.shape(x))
  if approximate:
    b = tf.abs(alpha - tf.cast(2., float_dtype)) + epsilon
    d = tf.where(tf.greater_equal(alpha, 0.), alpha + epsilon, alpha - epsilon)
    loss = (b / d) * (tf.pow(tf.square(x / scale) / b + 1., 0.5 * d) - 1.)
  else:
    squared_scaled_x = tf.square(x / scale)
    loss_two = 0.5 * squared_scaled_x
    loss_zero = log1p_safe(0.5 * squared_scaled_x)
    loss_neginf = -tf.math.expm1(-0.5 * squared_scaled_x)
    loss_posinf = expm1_safe(0.5 * squared_scaled_x)
    machine_epsilon = tf.cast(np.finfo(np.float32).eps, float_dtype)
    beta_safe = tf.maximum(machine_epsilon, tf.abs(alpha - 2.))
    alpha_safe = tf.where(tf.greater_equal(alpha, 0.), tf.ones_like(alpha), -tf.ones_like(alpha)) * tf.maximum(machine_epsilon, tf.abs(alpha))
    loss_otherwise = (beta_safe / alpha_safe) * (tf.pow(squared_scaled_x / beta_safe + 1., 0.5 * alpha) - 1.)
    loss = tf.where(tf.equal(alpha, -tf.cast(float('inf'), float_dtype)), loss_neginf, tf.where(tf.equal(alpha, 0.), loss_zero,tf.where(tf.equal(alpha, 2.), loss_two, tf.where(tf.equal(alpha, tf.cast(float('inf'), float_dtype)), loss_posinf, loss_otherwise))))
  return loss

def interpolate1d(x, values, tangents):
  float_dtype = x.dtype
  x_lo = tf.cast(tf.floor(tf.clip_by_value(x, 0., tf.cast(tf.shape(values)[0] - 2, float_dtype))), tf.int32)
  x_hi = x_lo + 1
  t = x - tf.cast(x_lo, float_dtype)
  t_sq = tf.square(t)
  t_cu = t * t_sq
  h01 = -2. * t_cu + 3. * t_sq
  h00 = 1. - h01
  h11 = t_cu - t_sq
  h10 = h11 - t_sq + t
  value_before = tangents[0] * t + values[0]
  value_after = tangents[-1] * (t - 1.) + values[-1]
  neighbor_values_lo = tf.gather(values, x_lo)
  neighbor_values_hi = tf.gather(values, x_hi)
  neighbor_tangents_lo = tf.gather(tangents, x_lo)
  neighbor_tangents_hi = tf.gather(tangents, x_hi)
  value_mid = (neighbor_values_lo * h00 + neighbor_values_hi * h01 + neighbor_tangents_lo * h10 + neighbor_tangents_hi * h11)
  return tf.where(t < 0., value_before, tf.where(t > 1., value_after, value_mid))

class Distribution(object):

  def __init__(self):
    with get_resource_as_file('partition_spline.npz') as spline_file:
      with np.load(spline_file, allow_pickle=False) as f:
        self._spline_x_scale = f['x_scale']
        self._spline_values = f['values']
        self._spline_tangents = f['tangents']

  def log_base_partition_function(self, alpha):
    float_dtype = alpha.dtype
    x = partition_spline_curve(alpha)
    return interpolate1d(x * tf.cast(self._spline_x_scale, float_dtype), tf.cast(self._spline_values, float_dtype), tf.cast(self._spline_tangents, float_dtype))

  def nllfun(self, x, alpha, scale):
    loss = lossfun(x, alpha, scale)
    log_partition = (tf.math.log(scale) + self.log_base_partition_function(alpha))
    nll = loss + log_partition
    return nll

  def draw_samples(self, alpha, scale):
    float_dtype = alpha.dtype
    shape = tf.shape(alpha)
    cauchy = tfp.distributions.Cauchy(loc=0., scale=tf.sqrt(2.))
    uniform = tfp.distributions.Uniform(low=0., high=1.)
    def while_cond(_, accepted):
      return ~tf.reduce_all(accepted)
    def while_body(samples, accepted):
      cauchy_sample = tf.cast(cauchy.sample(shape), float_dtype)
      nll = self.nllfun(cauchy_sample, alpha, tf.cast(1, float_dtype))
      nll_bound = lossfun(cauchy_sample, tf.cast(0, float_dtype), tf.cast(1, float_dtype), approximate=False) + self.log_base_partition_function(alpha)
      uniform_sample = tf.cast(uniform.sample(shape), float_dtype)
      accept = uniform_sample <= tf.math.exp(nll_bound - nll)
      samples = tf.where(accept, cauchy_sample, samples)
      accepted = accept | accepted
      return (samples, accepted)
    while_loop_vars = (tf.zeros(shape, float_dtype), tf.zeros(shape, dtype=bool))
    terminal_state = tf.while_loop(cond=while_cond, body=while_body, loop_vars=while_loop_vars)
    samples = tf.multiply(terminal_state[0], scale)
    return samples
	  
def _construct_scale(x, scale_lo, scale_init, float_dtype):
  if scale_lo == scale_init:
    scale = tf.tile(tf.cast(scale_init, float_dtype)[tf.newaxis, tf.newaxis], (1, x.shape[1]))
  else:
    latent_scale = tf.compat.v1.get_variable('LatentScale', initializer=tf.zeros((1, x.shape[1]), float_dtype))
    scale = affine_softplus(latent_scale, lo=scale_lo, ref=scale_init)
  return scale

class AdaptiveLossFunction(tf.Module):

  def __init__(self, num_channels, float_dtype=tf.float32, alpha_lo=0.001, alpha_hi=1.999, alpha_init=None, scale_lo=1e-5, scale_init=1.0, name=None):
    super(AdaptiveLossFunction, self).__init__(name=name)
    if alpha_lo != alpha_hi:
      if alpha_init is None: alpha_init = (alpha_lo + alpha_hi) / 2.
      latent_alpha_init = (inv_affine_sigmoid(alpha_init, lo=alpha_lo, hi=alpha_hi))
      self._latent_alpha = tf.Variable(tf.fill((1, num_channels), tf.cast(latent_alpha_init, dtype=float_dtype)), name='LatentAlpha')
    if scale_lo != scale_init: self._latent_scale = tf.Variable(tf.zeros((1, num_channels), float_dtype), name='LatentScale')
    self._num_channels = num_channels
    self._float_dtype = tf.dtypes.as_dtype(float_dtype)
    self._alpha_lo = alpha_lo
    self._alpha_hi = alpha_hi
    self._scale_lo = scale_lo
    self._scale_init = scale_init
    self._distribution = Distribution()

  def alpha(self):
    if self._alpha_lo == self._alpha_hi: return tf.tile(tf.cast(self._alpha_lo, self._float_dtype)[tf.newaxis, tf.newaxis], (1, self._num_channels))
    else: return affine_sigmoid(self._latent_alpha, lo=self._alpha_lo, hi=self._alpha_hi)

  def scale(self):
    if self._scale_lo == self._scale_init: return tf.tile(tf.cast(self._scale_init, self._float_dtype)[tf.newaxis, tf.newaxis], (1, self._num_channels))
    else: return affine_softplus(self._latent_scale, lo=self._scale_lo, ref=self._scale_init)

  def __call__(self, x):
    x = tf.convert_to_tensor(x)
    return self._distribution.nllfun(x, self.alpha(), self.scale())

class AdaptiveImageLossFunction(tf.Module):

  def __init__(self, image_size, float_dtype=tf.float32, summarize_loss=True, name=None, **kwargs):
    super(AdaptiveImageLossFunction, self).__init__(name=name)
    num_channels = np.prod(image_size)
    self._lossfun = AdaptiveLossFunction(num_channels, float_dtype, **kwargs)
    self._image_size = image_size
    self._float_dtype = tf.dtypes.as_dtype(float_dtype)
    self._summarize_loss = summarize_loss

  def alpha(self):
    return tf.reshape(self._lossfun.alpha(), self._image_size)

  def df(self):
    return tf.reshape(self._lossfun.df(), self._image_size)

  def scale(self):
    return tf.reshape(self._lossfun.scale(), self._image_size)

  def __call__(self, x):
    x = tf.convert_to_tensor(x)
    tf.debugging.assert_rank(x, 4)
    width, height, num_channels = self._image_size
    x_stack = tf.reshape(tf.transpose(x, perm=(0, 3, 1, 2)), (-1, width, height))
    x_mat = tf.reshape( tf.transpose( tf.reshape(x_stack, [-1, num_channels, width, height]), perm=[0, 2, 3, 1]), [-1, width * height * num_channels])
    loss_mat = self._lossfun(x_mat)
    loss = tf.reshape(loss_mat, [-1, width, height, num_channels])
    if self._summarize_loss:
      log_scale = tf.math.log(self.scale())
      log_scale_min = tf.reduce_min(log_scale)
      log_scale_max = tf.reduce_max(log_scale)
      tf.summary.image('robust/log_scale', (log_scale[tf.newaxis] - log_scale_min) / (log_scale_max - log_scale_min + 1e-10))
      tf.summary.histogram('robust/log_scale', log_scale)
      alpha = self.alpha()
      alpha_min = tf.reduce_min(alpha)
      alpha_max = tf.reduce_max(alpha)
      tf.summary.image('robust/alpha', (alpha[tf.newaxis] - alpha_min) / (alpha_max - alpha_min + 1e-10))
      tf.summary.histogram('robust/alpha', alpha)
    return loss

def averageinst(y_pred):
    y_pred[0] = tf.math.add(y_pred[0][:,:,:,:], y_pred[1][:,:,:,:])
    y_pred[0] = tf.math.divide(y_pred[0], tf.cast(2, tf.float32))
    return y_pred[0]
	
class CustomLossFunction(tf.Module):
  
  def __init__(self, width, length, targets, nsubgroups = [5, 12], nmodelpred = 1, reduce=False, name=None, **kwargs):
      super(CustomLossFunction, self).__init__(name=name)
      self.aux_l1 = AdaptiveImageLossFunction(image_size=(width,length,1))
      self.aux_l2 = AdaptiveImageLossFunction(image_size=(width,length,1))
      self.aux_l3 = AdaptiveImageLossFunction(image_size=(width,length,targets))
      self.nsubgroups = nsubgroups
      self.nmodelpred = nmodelpred
      self.reduce = reduce
  
  def __call__(self, y_true, y_pred):
        if self.nmodelpred == 2: y_pred_final = averageinst(y_pred)
        else: y_pred_final = y_pred
        numfinite = tf.math.count_nonzero(tf.math.is_finite(y_true[:,:,:,0]))
        mask = tf.where(tf.math.is_nan(y_true), tf.constant(0.0), tf.constant(1.0))
        y_true = tf.math.multiply_no_nan(y_true, mask)
        y_pred = tf.math.multiply_no_nan(y_pred, mask)

        # This should not be needed... in which circunstances are the predictions equal to nan? If they are, it's an error!
        # y_pred = tf.where(tf.math.is_nan(y_pred), K.constant(0), y_pred)

        # Custom loss sub-groups
        sumtrueg1 = tf.math.reduce_sum(y_true[:, :, :, 0:self.nsubgroups[0]], axis=3, keepdims=True)
        sumpredg1 = tf.math.reduce_sum(y_pred[:, :, :, 0:self.nsubgroups[0]], axis=3, keepdims=True)
        sumtrueg2 = tf.math.reduce_sum(y_true[:, :, :, self.nsubgroups[0]:self.nsubgroups[1]], axis=3, keepdims=True)
        sumpredg2 = tf.math.reduce_sum(y_pred[:, :, :, self.nsubgroups[0]:self.nsubgroups[1]], axis=3, keepdims=True)

        loss1 = self.aux_l1(sumtrueg1 - sumpredg1)
        loss2 = self.aux_l2(sumtrueg2 - sumpredg2)
        loss3 = tf.math.reduce_mean(self.aux_l3(y_true - y_pred_final), axis=3, keepdims=True)
        loss = loss1 + loss2 + loss3

        if self.reduce:
          loss = tf.math.divide_no_nan( tf.math.reduce_sum(loss) , tf.cast(numfinite, tf.float32) )
          max_val = np.finfo(np.float32).max
          loss = tf.clip_by_value(loss, -max_val, max_val)
        return loss
"""
batch_size = 1
width = 16
length = 16
targets = 12

ground_truth =  tf.random.uniform( ( batch_size, width, length, targets ) )
predictions = [ tf.random.uniform( ( batch_size, width, length, targets ) ) , tf.random.uniform( ( batch_size, width, length, targets ) ) ]
predictions = averageinst( predictions )
mylossfun = CustomLossFunction(width, length, targets)
loss = mylossfun(predictions, ground_truth)

print( ground_truth.shape )
print( predictions.shape )
print( loss.shape )
print( len(list(mylossfun.trainable_variables)) )
"""