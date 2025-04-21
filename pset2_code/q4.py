import numpy as np

##### Part b  - function #####
def lstsq_cv_err(features: np.ndarray, labels: np.ndarray, subset_count: int = 8) -> float:
    """Estimate the error of a least-squares classifier using cross-validation. Use subset_count different train/test splits with each subset acting as the holdout set once.
        
    Parameters:
        features (np.ndarray): dataset features as a 2D array with shape (sample_count , feature_count) 
        labels (np.ndarray): dataset class labels (+1/-1) as a 1D array with length (sample_count) 
        subset_count (int): number of subsets to divide the dataset into
            Note: assumes that subset_count divides the dataset evenly
        
    Returns:
    cls_err (float): estimated classification error rate of least-squares method"""

    sample_count, feature_count = features.shape
    subset_size = sample_count // subset_count

    # Reshape arrays for easier subset-level manipulation 
    reshaped_feat = features.reshape(subset_count, subset_size, feature_count)
    reshaped_lbls = labels.reshape(subset_count, subset_size)

    subset_idcs = np.arange(subset_count)
    train_set_size = (subset_count - 1) * subset_size 
    subset_err_counts = np.zeros(subset_count)

    for i in range(subset_count):
        # TODO: select relevant dataset, 
        # fit and evaluate a linear model,
        # then store errors in subset_err_counts[i]
        # Hint: you could extract the training subset with train_subset_idcs = subset_idcs[subset_idcs != i]

        test_feat = reshaped_feat[i]
        test_label = reshaped_lbls[i]

        train_subset_idcs = subset_idcs[subset_idcs != i]
        train_feat = reshaped_feat[train_subset_idcs].reshape(train_set_size, feature_count)
        train_label = reshaped_lbls[train_subset_idcs].reshape(train_set_size)

        XT_X = np.dot(train_feat.T, train_feat)
        XT_y = np.dot(train_feat.T, train_label)
        w = np.dot(np.linalg.inv(XT_X), XT_y)

        y_pred = np.sign(np.dot(test_feat, w))
        subset_err_counts[i] = np.sum(y_pred != test_label)

    # Average over the entire dataset to find the classification error
    cls_err = np.sum(subset_err_counts) / (subset_count * subset_size)
    return cls_err


##### Part e - function #####
def drop_features_heuristic(features, labels, max_cv_err=0.06):
    chosen_features = list(range(features.shape[1]))
    best_err = lstsq_cv_err(features, labels)

    while len(chosen_features) > 1:
        reducedX = features[:, chosen_features]
        XT_X = reducedX.T @ reducedX
        XT_y = reducedX.T @ labels
        w = np.linalg.inv(XT_X) @ XT_y

        # absolute value weights, sorted from least to most 
        min_idx = np.argmin(np.abs(w))
        feat_to_drop = chosen_features[min_idx]

        trial_features = [f for f in chosen_features if f != feat_to_drop]
        trialX = features[:, trial_features]
        trial_err = lstsq_cv_err(trialX, labels)

        if trial_err <= max_cv_err:
            chosen_features = trial_features
            best_err = trial_err
            print(f"Part␣4e.␣dropping feature {feat_to_drop}, error: {trial_err*100:.2f}%")
        else:
            print(f"Part␣4e.␣dropping feature {feat_to_drop}, error {trial_err*100:.2f}% > {max_cv_err*100:.2f}%")
            break

    return chosen_features, best_err

##### Part a - main #####
# Load in training data and labels
# File available on Canvas

face_data_dict = np.load("face_emotion_data.npz") 
face_features = face_data_dict["X"]
face_labels = face_data_dict["y"]
n, p = face_features.shape

# Solve the least-squares solution. weights is the array of weight coefficients
# TODO: find weights
XT_X = np.dot(face_features.T, face_features)          # X^T X
XT_y = np.dot(face_features.T, face_labels)          # X^T y
weights = np.dot(np.linalg.inv(XT_X), XT_y)  # (X^T X)^{-1} X^T y

print(f"Part␣4a.␣Found␣weights:\n{weights}")

##### Part b - main #####
# Run on the dataset with all features included 
full_feat_cv_err = lstsq_cv_err(face_features , face_labels)
print(f"Part␣4b.␣Error␣estimate:␣{full_feat_cv_err*100:.3f}%")

##### Part e - main #####
chosen_features, final_err =  drop_features_heuristic(face_features, face_labels, max_cv_err=0.06)
print("Part␣4e.␣Select␣features:", chosen_features)
print(f"Part␣4e.␣CV␣error␣with␣select␣features:␣{final_err*100:.2f}%")



