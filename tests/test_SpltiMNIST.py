from cont_flame.dataset import SplitMNIST

def test_iterate():
    ds = SplitMNIST()
    for x in iter(ds):
        pass

test_iterate()