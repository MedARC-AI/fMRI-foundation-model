from util.hcp_flat import create_hcp_flat


def test_create_hcp_flat():
    dataset = create_hcp_flat()
    key, img = next(iter(dataset))
    print(key, img.shape)
