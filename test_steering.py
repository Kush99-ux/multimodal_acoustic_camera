from acoustic.steering import SteeringFactory

factory = SteeringFactory()

print("Microphone geometry:")
print(factory.get_microphone_geometry())

print()

print("Grid shape:", factory.grid_shape())

print()

print("Frequencies:")
print(factory.get_frequencies())

print()

print("Steering vector object:")
print(factory.get_steering_vector())