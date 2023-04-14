import numpy as np
import pdb


def get_overlapping_default_events(window_size=None, default_event_sizes=None, factor_overlap=None):

    default_events = []
    for default_event_size in default_event_sizes:
        overlap = default_event_size / factor_overlap
        number_of_default_events = int(window_size / overlap)
        default_events.extend(
            [
                (overlap * (0.5 + i) / window_size, default_event_size / window_size)
                for i in range(number_of_default_events)
            ]  # TODO: I don't feel this is correctly implemented. It should start from default_event_size // factor_overlap
        )

    return np.array(default_events, dtype=np.float32)


def match_events_localization_to_default_localizations(
    self, _localizations_default, events, threshold_overlap, window
):
    if not isinstance(events, list):
        events = [events]
    batch = len(events)
    repeat_factor = _localizations_default.shape[0] // window

    # Find localizations_target and classifications_target by matching
    # ground truth localizations to default localizations
    if repeat_factor > 0:
        localizations_default = _localizations_default[:repeat_factor, :]
    else:
        localizations_default = _localizations_default
    number_of_default_events = localizations_default.shape[0]
    localizations_target = np.zeros([batch, number_of_default_events, 2], dtype=np.float32)
    classifications_target = np.zeros([batch, number_of_default_events], dtype=np.float32)

    for batch_index in range(batch):

        # If no event add default value to predict (will never be used anyway)
        # And class 0 == background
        if events[batch_index].size == 0:
            localizations_target[batch_index][:, :] = np.tile(np.array([[-1, 1]]), [localizations_default.shape[0], 1])
            classifications_target[batch_index] = np.zeros(localizations_default.shape[0])
            continue

        # Else match to most overlapping event and set to background depending on threshold
        localizations_truth = events[batch_index][:, :2]
        classifications_truth = events[batch_index][:, -1]
        localizations_a = localizations_truth
        localizations_b = np.concatenate(
            (
                (localizations_default[:, 0] - localizations_default[:, 1] / 2)[:, None],
                (localizations_default[:, 0] + localizations_default[:, 1] / 2)[:, None],
            ),
            axis=1,
        )
        overlaps = jaccard_overlap(localizations_a, localizations_b)

        # (Bipartite Matching) https://github.com/amdegroot/ssd.pytorch/blob/master/ssd.py
        # might be usefull if an event is included in another
        best_prior_index = overlaps.argmax(1)
        best_truth_overlap, best_truth_idx = overlaps.max(0), overlaps.argmax(0)

        # ensure every gt matches with its prior of max overlap
        np.put_along_axis(best_truth_overlap, best_prior_index, values=2, axis=0)
        for j in range(best_prior_index.shape[0]):
            best_truth_idx[best_prior_index[j]] = j

        localization_match = localizations_truth[best_truth_idx]
        localization_target = encode(localization_match, localizations_default)
        classification_target = classifications_truth[best_truth_idx]  # + 1  # Add class 0!
        classification_target[best_truth_overlap < threshold_overlap] = 0

        localizations_target[batch_index][:, :] = localization_target
        classifications_target[batch_index] = classification_target

    if repeat_factor > 0:
        return np.tile(localizations_target, [1, window, 1]), np.tile(classifications_target, [1, window])
    else:
        return localizations_target, classifications_target


def encode(localization_match, localizations_default):
    """localization_match are converted relatively to their default location
    localization_match has size [batch, number_of_localizations, 2] containing the ground truth
    matched localization (representation x y)
    localization_defaults has size [number_of_localizations, 2]
    returns localization_target [batch, number_of_localizations, 2]
    """
    center = (localization_match[:, 0] + localization_match[:, 1]) / 2 - localizations_default[:, 0]
    center = center / localizations_default[:, 1]
    if (localization_match[:, 1] - localization_match[:, 0] == 0).any():
        breakpoint()
    width = np.log((localization_match[:, 1] - localization_match[:, 0]) / localizations_default[:, 1])
    localization_target = np.concatenate([center[:, None], width[:, None]], 1)
    # width = torch.log((localization_match[:, 1] - localization_match[:, 0]) / localizations_default[:, 1])
    # localization_target = torch.cat([center.unsqueeze(1), width.unsqueeze(1)], 1)
    return localization_target


def jaccard_overlap(localizations_a, localizations_b):
    """Jaccard overlap between two segments A ∩ B / (LENGTH_A + LENGTH_B - A ∩ B)

    localizations_a: tensor of localizations
    localizations_a: tensor of localizations
    """
    A = localizations_a.shape[0]
    B = localizations_b.shape[0]
    max_min = np.maximum(
        np.tile(localizations_a[:, 0][:, None], [1, B]), np.tile(localizations_b[:, 0][None, :], [A, 1])
    )
    min_max = np.minimum(
        np.tile(localizations_a[:, 1][:, None], [1, B]), np.tile(localizations_b[:, 1][None, :], [A, 1])
    )
    intersection = np.clip(min_max - max_min, a_min=0, a_max=None)
    length_a = np.tile((localizations_a[:, 1] - localizations_a[:, 0])[:, None], [1, B])
    length_b = np.tile((localizations_b[:, 1] - localizations_b[:, 0])[None, :], [A, 1])
    if (length_a + length_b - intersection == 0).any():
        pdb.set_trace()
    try:
        overlaps = intersection / (length_a + length_b - intersection)
    except RuntimeWarning:
        pdb.set_trace()

    return overlaps
