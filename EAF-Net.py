
def EAF_res_module(self, feature, xyz,rgb, neigh_idx, d_out, name, is_training):
    f_pc = helper_tf_util.conv2d(feature, d_out // 2, [1, 1], name + 'mlp1', [1, 1], 'VALID', True, is_training)
    f_pc = self.EAF_module(xyz,rgb, f_pc, neigh_idx, d_out, name + 'EAF', is_training)
    f_pc = helper_tf_util.conv2d(f_pc, d_out * 2, [1, 1], name + 'mlp2', [1, 1], 'VALID', True, is_training,
                                    activation_fn=None)

    shortcut = helper_tf_util.conv2d(feature, d_out * 2, [1, 1], name + 'shortcut', [1, 1], 'VALID',
                                        activation_fn=None, bn=True, is_training=is_training)
    return tf.nn.leaky_relu(f_pc + shortcut)

def EAF_module(self, xyz,rgb,feature, neigh_idx, d_out, name, is_training):
    d_in = feature.get_shape()[-1].value

    f_info = self.information_encoding(xyz,rgb,feature,neigh_idx)
    f_info = tf.nn.leaky_relu(tf.layers.batch_normalization(
        f_info, -1, 0.99, 1e-6, training=is_training
    ))
    f_info = helper_tf_util.conv2d(
        f_info, d_in, [1, 1], name + 'mlp1', [1, 1], 'VALID', True, is_training,
        activation_fn=tf.nn.leaky_relu
    )
    f_info = helper_tf_util.conv2d(
        f_info, d_in, [1, 1], name + 'mlp2', [1, 1], 'VALID', False, is_training,
        activation_fn=None
    )

    f_agg = self.external_attention(feature,name + 'ext_att', is_training)
    f_agg = tf.nn.leaky_relu(tf.layers.batch_normalization(
        f_agg, -1, 0.99, 1e-6, training=is_training
    ))

    f_neighbours = self.gather_neighbour(f_agg, neigh_idx)
    f_concat = tf.concat([f_neighbours, f_info], axis=-1)
    f_out = self.att_pooling(f_concat, d_out, name + 'att_pooling_1', is_training)

    return f_out

def information_encoding(self, xyz,rgb,feature,neigh_idx):
    neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)
    xyz_tile = tf.tile(tf.expand_dims(xyz, axis=2), [1, 1, tf.shape(neigh_idx)[-1], 1])
    relative_xyz = xyz_tile - neighbor_xyz
    relative_dis = tf.sqrt(tf.reduce_sum(tf.square(relative_xyz), axis=-1, keepdims=True))
    neighbor_rgb = self.gather_neighbour(rgb, neigh_idx)
    rgb_tile = tf.tile(tf.expand_dims(rgb, axis=2), [1, 1, tf.shape(neigh_idx)[-1], 1])
    relative_rgb = rgb_tile - neighbor_rgb
    neighbor_feature = self.gather_neighbour(tf.squeeze(feature, axis=2), neigh_idx)
    feature_tile = tf.tile(feature, [1, 1, tf.shape(neigh_idx)[-1], 1])
    
    diff_i = feature_tile  
    diff_j = neighbor_feature  
    fi = tf.nn.l2_normalize(diff_i, axis=-1)
    fj = tf.nn.l2_normalize(diff_j, axis=-1)

    cos_sim = tf.reduce_sum(fi * fj, axis=-1, keepdims=True)  
    semantic_dis = 1.0 - cos_sim 
    relative_feature = tf.concat([relative_xyz, relative_dis,relative_rgb,semantic_dis], axis=-1)

    return relative_feature

def external_attention(self, feature, name, is_training=True, num_heads=4):

    d_in = feature.shape[-1].value
    mem_num = 64

    if d_in is None or d_in % num_heads != 0:
        num_heads = 1
    head_dim = d_in // num_heads

    feature = tf.squeeze(feature, axis=2)  # [B, N, d_in]
    B = tf.shape(feature)[0]
    N = tf.shape(feature)[1]

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        F = tf.layers.dense(feature, d_in, activation=tf.nn.relu, name="linear_transform")
        F = tf.reshape(F, [B, N, num_heads, head_dim])
        F = tf.transpose(F, [0, 2, 1, 3])  # [B, H, N, C]

        M_k = tf.get_variable(
            'M_k',
            [num_heads, mem_num, head_dim],
            initializer=tf.initializers.glorot_normal()
        )
        M_v = tf.get_variable(
            'M_v',
            [num_heads, mem_num, head_dim],
            initializer=tf.initializers.glorot_normal()
        )

        attn = tf.einsum('bhnc,hcm->bhnm', F, tf.transpose(M_k, [0, 2, 1]))
        attn = tf.nn.softmax(attn, axis=3)

        l1_norm = tf.reduce_sum(tf.abs(attn), axis=3, keepdims=True) + 1e-8
        attn = attn / l1_norm

        out = tf.einsum('bhnm,hmc->bhnc', attn, M_v)
        out = tf.transpose(out, [0, 2, 1, 3])  # [B, N, H, C]
        out = tf.reshape(out, [B, N, d_in])

        out_flat = out + feature
        return out_flat

def att_pooling(feature_set, d_out, name, is_training):
    with tf.variable_scope(name):
        d = feature_set.get_shape()[3].value
        att_scores = helper_tf_util.conv2d(
            feature_set,
            d, 
            [1, 1],
            name + "_att_conv",
            [1, 1],
            "VALID",
            bn=True,
            is_training=is_training,
            activation_fn=None  
        )

        att_scores = tf.nn.softmax(att_scores, axis=2)
        weighted_features = feature_set * att_scores
        f_agg = tf.reduce_sum(weighted_features, axis=2, keepdims=True)

        f_out = helper_tf_util.conv2d(
            f_agg,
            d_out,
            [1, 1],
            name + "_out_conv",
            [1, 1],
            "VALID",
            bn=True,
            is_training=is_training
        )

        return f_out