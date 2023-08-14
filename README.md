# eucaim_dl_model

Read before using:

 1. Download the full images dataset from [kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
 2. Prepare at least 2 _data-sites_ using the _.csv_ files from [this repo](https://github.com/EUCAIM/demo_dl_data/tree/main/data_ids)
   * Each site must go inside the main `chest_xray` and follow: `test_N`, `train_N`, and `val_N` were `N` is the number of the _data-site_ (1, 2, 3...)
 3. Raise the server
 4. Raise each client, modifiing the `n_client` valirable each time one is raised so the proper _data-set_ is used