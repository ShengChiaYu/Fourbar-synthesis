def select_test_image(model, testset, args, index):
    model.eval()
    num, image, label, params = testset.__getitem__(index)
    print('Test image number: ' + str(num))

    image = image.unsqueeze(0).cuda(args.gpu_id)
    test_feature = model(image, feature=True)

    return test_feature


def path_plot(testset, index, trainset, retrieved_ind):
    _, _, _, t_params = testset.__getitem__(index)
    _, _, _, r_params = trainset.__getitem__(retrieved_ind)

    n = 512
    p1, p2 = path_gen_open(t_params[:4], t_params[4], t_params[5], t_params[6], n, t_params[7], t_params[8])
    if index < testset.len/2:
        t_points = p1
    else:
        t_points = p2
    p1, p2 = path_gen_open(r_params[:4], r_params[4], r_params[5], r_params[6], n, r_params[7], r_params[8])
    if retrieved_ind < trainset.len/2:
        r_points = p1
    else:
        r_points = p2

    plt.plot(t_points[:,0],t_points[:,1],'ro', markersize=1)
    plt.plot(r_points[:,0],r_points[:,1],'bo', markersize=1)
    plt.plot(0,0,'k+')
    plt.axis('equal')
    plt.show()


def cosine_similarity(features, test_feature):
    dot_product = (features*test_feature).sum(dim=1, keepdim=True)
    # print(dot_product.shape)
    features_norm = features.norm(p=2, dim=1, keepdim=True)
    # print(features_norm.shape)
    test_feature_norm = test_feature.norm(p=2, dim=1, keepdim=True)
    # print(test_feature_norm.shape)
    cos = dot_product / (features_norm*test_feature_norm)
    # print(cos.shape)

    return cos
