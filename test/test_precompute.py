from acoustic.steering import SteeringFactory

factory = SteeringFactory()

print("Grid shape:", factory.grid_shape())
print("Number of frequencies:", len(factory.get_frequencies()))

factory.precompute_matrices()

print("Cache size:", factory.cache_size())

# Test one frequency
A = factory.get_matrix(1000.0)

print("Steering matrix shape:", A.shape)
print("Matrix dtype:", A.dtype)