import torch


def collate(batch):
    if "classifications_target" not in batch[0].keys():

        batch_eeg = torch.stack([torch.Tensor(el["signal"]) for el in batch])
        batch_events = [el["events"] for el in batch]
        batch_records = [el["record"] for el in batch]

        return batch_eeg, batch_events, batch_records

    else:

        batch_data = torch.stack([torch.Tensor(el["signal"]) for el in batch])
        batch_events = [torch.Tensor(el["events"]) for el in batch]
        batch_records = [el["record"] for el in batch]
        batch_clf = torch.stack([torch.LongTensor(el["classifications_target"]) for el in batch])
        batch_loc = torch.stack([torch.Tensor(el["localizations_target"]) for el in batch])
        batch_stages = torch.stack([torch.from_numpy(el["stages"]) for el in batch])

        return batch_data, batch_events, batch_records, batch_clf, batch_loc, batch_stages
