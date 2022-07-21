import torch

import torch.nn.functional as F


def get_loss_func(content_feature_maps, style_feature_maps,
                  content_feature_maps_idx, style_feature_maps_indices,
                  alpha, beta):

    style_gramm_matrices = [gramm_matrix(x) for i, x in enumerate(style_feature_maps) if i in style_feature_maps_indices]

    # keep in mind that this function could be more optimal,
    # however in this form it is easier to follow along with the paper
    def loss(target_feature_maps):
        content_loss = .5 * F.mse_loss(target_feature_maps[content_feature_maps_idx],
                                       content_feature_maps[content_feature_maps_idx],
                                       reduction='sum')

        style_loss = 0
        for i, idx in enumerate(style_feature_maps_indices):
            channel_num, height, width = target_feature_maps[idx].shape
            G = gramm_matrix(target_feature_maps[idx])
            A = style_gramm_matrices[i]

            # contribution of a single layer to the style loss
            E = F.mse_loss(G, A, reduction='sum') / \
                (2*channel_num*height*width)**2

            # according to the paper weighting factor of each layer used
            # was equal to 1 divided by the number of acitve layers with non-zero loss-weight
            style_loss += E / len(style_feature_maps_indices)

        total_loss = alpha*content_loss + beta*style_loss

        return total_loss, content_loss, style_loss

    return loss


def gramm_matrix(x):
    channel_num, height, width = x.shape
    unrolled = x.view(channel_num, height*width)
    return torch.mm(unrolled, unrolled.t())
