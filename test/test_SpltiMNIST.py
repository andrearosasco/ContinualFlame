from dataset.SplitMNIST import SplitMNIST


def test_iterate():
    ds = SplitMNIST()
    for x in iter(ds):
        pass
