# Add Value Loss as a Training Regularizer

Value-JEPA LeWM adds a ValueJEPA-style expectile TD value loss only to the training policy, while deliberately keeping the JEPA model structure, planner, and evaluation latent cost unchanged. This makes the first experiment a clean test of whether value-shaped latent geometry improves LeWM, without introducing a value head, quasimetric module, or planner-side scoring change.
