

def aggregate_minmax(results):
    """Compute weighted average or max/min based on majority."""
    weighted_weights = [weights for weights, _ in results]

    num_layers = len(weighted_weights[0])
    weights_prime = []
    for layer_idx in range(num_layers):
        stacked_layer = np.stack([weights[layer_idx] for weights in weighted_weights], axis=0)

        pos_count = np.sum(stacked_layer > 0, axis=0)
        neg_count = np.sum(stacked_layer < 0, axis=0)

        max_values = np.max(stacked_layer, axis=0)
        min_values = np.min(stacked_layer, axis=0)
        mean_values = np.mean(stacked_layer, axis=0)

        aggregated = np.where(
            pos_count > neg_count,  
            max_values,
            np.where(
                neg_count > pos_count,  
                min_values,
                mean_values, 
            ),
        )
        weights_prime.append(aggregated)
    return weights_prime


def aggregate_maxdiff(results, global_model):
    """Select model with highest difference compared to global model."""

    global_model = [np.array(layer) for layer in global_model]

        # Calculate the differences between each model's weights and the global model's weights
    differences = []
    for i, (weights, _) in enumerate(results):
        weights = [np.array(layer) for layer in weights]
        diff = sum(np.linalg.norm(layer - global_layer) for layer, global_layer in zip(weights, global_model))
        differences.append((i, diff))

    # Find the model with the highest difference
    max_diff_model_index = max(differences, key=lambda x: x[1])[0]

    # Use the weights of the model with the highest difference
    weights_prime = results[max_diff_model_index][0]

    return weights_prime



def aggregate_mindiff(results, global_model):
    """Select model with lowest difference compared to global model."""

    global_model = [np.array(layer) for layer in global_model]

        # Calculate the differences between each model's weights and the global model's weights
    differences = []
    for i, (weights, _) in enumerate(results):
        weights = [np.array(layer) for layer in weights]
        diff = sum(np.linalg.norm(layer - global_layer) for layer, global_layer in zip(weights, global_model))
        differences.append((i, diff))

    # Find the model with the lowest difference
    max_diff_model_index = min(differences, key=lambda x: x[1])[0]

    # Use the weights of the model with the lowest difference
    weights_prime = results[max_diff_model_index][0]

    return weights_prime


def aggregate_median(results):
    """Compute weighted median."""
    weighted_weights = [
        [layer for layer in weights] for weights, _ in results
    ]
    weights_prime = [
        np.median(np.array(layer_updates), axis=0)
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime
