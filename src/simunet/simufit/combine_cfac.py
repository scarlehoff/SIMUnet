from n3fit.backends import MetaLayer
from simunet.smeft import parameter_responses


def _choose_initializer(ini_dict):
    """Create an initializer whose values are physical coefficients."""
    if ini_dict["type"] == "uniform":
        max_v = ini_dict["maxval"]
        min_v = ini_dict["minval"]
        return MetaLayer.select_initializer("random_uniform", minval=min_v, maxval=max_v)
    elif ini_dict["type"] == "constant":
        return MetaLayer.init_constant(ini_dict["value"])
    raise ValueError(f"{ini_dict} not understood")


class CombineCfacLayer(MetaLayer):

    def __init__(self, simu_parameters, name="SimunetFactor", **kwargs):
        self._simu_parameters = simu_parameters
        self._kernel = []
        super().__init__(name=name, **kwargs)

    def apply_linear_comb(self, cfactors=None):
        """Take all cfactors and returns a list of pre-computed values to call this function with."""
        return parameter_responses(self._simu_parameters, cfactors)

    @property
    def scales(self):
        """Return the scales in self._simu_parameters as an iterable."""
        for parameter in self._simu_parameters:
            yield parameter.get("scale", 1.0)

    def build(self, input_shape):
        """Build stage should only be run at compile time or first inference."""
        for parameter in self._simu_parameters:
            initializer = _choose_initializer(parameter["initialisation"])
            scale = parameter.get("scale", 1.0)

            # Keras calls this with the weight shape; scale the resulting tensor.
            # Bind defaults so each parameter retains its own initializer and scale.
            def scaled_initializer(shape, dtype=None, initializer=initializer, scale=scale):
                return initializer(shape, dtype=dtype) * scale

            ker = self.builder_helper(
                name=parameter["name"],
                kernel_shape=(1,),  # TODO here we could have a different one per replica
                initializer=scaled_initializer,
                trainable=True,
            )
            self._kernel.append(ker)
        super().build(input_shape)

    def call(self, linear_comb, observables):
        wsum = 0.0
        for kernel, arr in zip(self._kernel, linear_comb):
            wsum += kernel * arr
        return (1.0 + wsum) * observables
