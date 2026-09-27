"""JAX spatio-temporal graph encoder used by diff_cvar_gnn."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from model.jax_ppo_base import activation, dense, init_head, init_linear, mlp


def _init_mlp(keys, dims, gains, use_init_weights=True):
    return [
        init_linear(key, in_dim, out_dim, gain, use_init_weights)
        for key, in_dim, out_dim, gain in zip(keys, dims[:-1], dims[1:], gains)
    ]


def init_reference_attention_layer(key, node_dim, edge_dim, hidden_dim, output_dim, use_init_weights=True):
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    pair_dim = 2 * int(node_dim) + int(edge_dim)
    sqrt2 = jnp.sqrt(jnp.asarray(2.0, dtype=jnp.float32))
    return {
        "psi1": _init_mlp(
            (k1, k2),
            (pair_dim, int(hidden_dim), int(output_dim)),
            (float(sqrt2), float(sqrt2)),
            use_init_weights,
        ),
        "psi2": _init_mlp(
            (k3, k4),
            (int(output_dim), int(output_dim), 1),
            (float(sqrt2), 1.0),
            use_init_weights,
        ),
        "psi3": _init_mlp(
            (k5, k6),
            (int(output_dim), int(hidden_dim), int(output_dim)),
            (float(sqrt2), float(sqrt2)),
            use_init_weights,
        ),
    }


def _dense_flat(layer, x):
    return _dense_flat_no_bias(layer["weight"], x) + layer["bias"]


def _dense_flat_no_bias(weight, x):
    leading_shape = x.shape[:-1]
    flat_x = x.reshape((-1, x.shape[-1]))
    flat_out = jnp.sum(flat_x[:, None, :] * weight[None, :, :], axis=-1)
    return flat_out.reshape((*leading_shape, weight.shape[0]))


def _edge_mlp(layers, x, act_name):
    for layer in layers[:-1]:
        x = activation(_dense_flat(layer, x), act_name)
    return _dense_flat(layers[-1], x)


def _spatio_temporal_edge_indices(history_len, num_entities):
    receivers = []
    senders = []
    edge_types = []
    history_len = int(history_len)
    num_entities = int(num_entities)
    for t in range(history_len):
        base = t * num_entities
        for receiver_entity in range(num_entities):
            for sender_entity in range(num_entities):
                if receiver_entity == sender_entity:
                    continue
                receivers.append(base + receiver_entity)
                senders.append(base + sender_entity)
                edge_types.append(0.0)
    for t in range(history_len - 1):
        base = t * num_entities
        next_base = (t + 1) * num_entities
        for entity in range(num_entities):
            receivers.append(base + entity)
            senders.append(next_base + entity)
            edge_types.append(1.0)
            receivers.append(next_base + entity)
            senders.append(base + entity)
            edge_types.append(1.0)
    return (
        jnp.asarray(receivers, dtype=jnp.int32),
        jnp.asarray(senders, dtype=jnp.int32),
        jnp.asarray(edge_types, dtype=jnp.float32),
    )


def _sparse_edge_mlp_from_parts(layers, nodes, edges, receiver_idx, sender_idx, act_name):
    first = layers[0]
    node_dim = nodes.shape[-1]
    weight = first["weight"]
    recv_weight = weight[:, :node_dim]
    send_weight = weight[:, node_dim : 2 * node_dim]
    edge_weight = weight[:, 2 * node_dim :]
    receiver_nodes = nodes[:, receiver_idx, :]
    sender_nodes = nodes[:, sender_idx, :]
    x = (
        _dense_flat_no_bias(recv_weight, receiver_nodes)
        + _dense_flat_no_bias(send_weight, sender_nodes)
        + _dense_flat_no_bias(edge_weight, edges)
        + first["bias"]
    )
    x = activation(x, act_name)
    for layer in layers[1:-1]:
        x = activation(_dense_flat(layer, x), act_name)
    return _dense_flat(layers[-1], x)


def reference_attention_layer_sparse(params, nodes, edges, node_mask, receiver_idx, sender_idx, act_name="relu"):
    batch_size, num_nodes, _node_dim = nodes.shape
    q_edges = _sparse_edge_mlp_from_parts(params["psi1"], nodes, edges, receiver_idx, sender_idx, act_name)
    scores = _edge_mlp(params["psi2"], q_edges, act_name).squeeze(-1)

    edge_valid = node_mask[:, receiver_idx] & node_mask[:, sender_idx]
    scores = jnp.where(edge_valid, scores, jnp.asarray(-1.0e9, dtype=scores.dtype))
    max_per_receiver = jnp.full((batch_size, num_nodes), -1.0e9, dtype=scores.dtype)
    max_per_receiver = max_per_receiver.at[:, receiver_idx].max(scores)
    shifted = scores - max_per_receiver[:, receiver_idx]
    exp_scores = jnp.exp(shifted) * edge_valid.astype(scores.dtype)
    sum_per_receiver = jnp.zeros((batch_size, num_nodes), dtype=scores.dtype)
    sum_per_receiver = sum_per_receiver.at[:, receiver_idx].add(exp_scores)
    denom = sum_per_receiver[:, receiver_idx]
    weights = jnp.where(denom > 0.0, exp_scores / jnp.maximum(denom, 1.0e-12), 0.0)

    messages = _edge_mlp(params["psi3"], q_edges, act_name)
    node_emb = jnp.zeros((batch_size, num_nodes, messages.shape[-1]), dtype=messages.dtype)
    node_emb = node_emb.at[:, receiver_idx, :].add(weights[..., None] * messages)
    return node_emb * node_mask[..., None].astype(node_emb.dtype)

def init_spatio_temporal_encoder(
    key,
    obs_frame_dim,
    graph_hidden_dim=64,
    graph_latent_dim=16,
    graph_layers=3,
    use_init_weights=True,
):
    obs_frame_dim = int(obs_frame_dim)
    if obs_frame_dim < 6 or (obs_frame_dim - 6) % 6 != 0:
        raise ValueError("SpatioTemporalGraphObservationEncoder expects obs_frame_dim = 6 + num_obstacles * 6")
    graph_layers = int(graph_layers)
    if graph_layers <= 0:
        raise ValueError("graph_layers must be > 0")
    keys = jax.random.split(key, graph_layers)
    layers = []
    node_dim = 3
    for layer_key in keys:
        layers.append(
            init_reference_attention_layer(
                layer_key,
                node_dim=node_dim,
                edge_dim=6,
                hidden_dim=int(graph_hidden_dim),
                output_dim=int(graph_latent_dim),
                use_init_weights=use_init_weights,
            )
        )
        node_dim = int(graph_latent_dim)
    return {"layers": layers}


def _reshape_obs_history(obs_history, history_len, obs_frame_dim):
    obs_history = jnp.asarray(obs_history, dtype=jnp.float32)
    if obs_history.ndim == 2:
        return obs_history.reshape((obs_history.shape[0], int(history_len), int(obs_frame_dim)))
    if obs_history.ndim == 3:
        return obs_history
    raise ValueError(f"Expected obs_history ndim 2 or 3, got {obs_history.ndim}")


def _world_to_body(vectors, robot_theta):
    c = jnp.cos(robot_theta)
    s = jnp.sin(robot_theta)
    while c.ndim < vectors.ndim - 1:
        c = c[..., None]
        s = s[..., None]
    x = vectors[..., 0]
    y = vectors[..., 1]
    return jnp.stack([c * x + s * y, -s * x + c * y], axis=-1)


def _build_spatio_temporal_graph_tensors(hparams, obs_history):
    obs_history = _reshape_obs_history(obs_history, hparams.history_len, hparams.obs_frame_dim)
    batch_size, history_len, _ = obs_history.shape
    num_obstacles = int(hparams.num_humans)
    num_entities = num_obstacles + 2
    total_nodes = int(history_len) * num_entities
    dtype = obs_history.dtype

    goal_rel = obs_history[:, :, 0:2]
    robot_vel = obs_history[:, :, 2:4]
    robot_theta = obs_history[:, :, 4]
    robot_radius = obs_history[:, :, 5:6]

    obstacle_blocks = obs_history[:, :, 6:].reshape((batch_size, history_len, num_obstacles, 6))
    robot_minus_obstacle = obstacle_blocks[:, :, :, 0:2]
    obstacle_vel = obstacle_blocks[:, :, :, 2:4]
    obstacle_radius = obstacle_blocks[:, :, :, 4:5]
    obstacle_mask = jnp.clip(obstacle_blocks[:, :, :, 5], 0.0, 1.0) > 0.5

    robot_pos = jnp.zeros((batch_size, history_len, 1, 2), dtype=dtype)
    obstacle_pos = -robot_minus_obstacle
    goal_pos = -goal_rel[:, :, None, :]
    positions = jnp.concatenate([robot_pos, obstacle_pos, goal_pos], axis=2)

    goal_vel = jnp.zeros((batch_size, history_len, 1, 2), dtype=dtype)
    velocities = jnp.concatenate([robot_vel[:, :, None, :], obstacle_vel, goal_vel], axis=2)

    goal_radius = jnp.zeros((batch_size, history_len, 1, 1), dtype=dtype)
    radii = jnp.concatenate([robot_radius[:, :, None, :], obstacle_radius, goal_radius], axis=2)

    positions = _world_to_body(positions, robot_theta)
    velocities = _world_to_body(velocities, robot_theta)

    node_types = jnp.zeros((batch_size, history_len, num_entities, 3), dtype=dtype)
    node_types = node_types.at[:, :, 0, 0].set(1.0)
    node_types = node_types.at[:, :, 1:-1, 1].set(1.0)
    node_types = node_types.at[:, :, -1, 2].set(1.0)

    robot_goal_mask = jnp.ones((batch_size, history_len, 1), dtype=bool)
    node_mask = jnp.concatenate([robot_goal_mask, obstacle_mask, robot_goal_mask], axis=2)
    nodes = node_types.reshape((batch_size, total_nodes, 3))
    node_mask_flat = node_mask.reshape((batch_size, total_nodes))
    nodes = nodes * node_mask_flat[..., None].astype(dtype)

    positions_flat = positions.reshape((batch_size, total_nodes, 2))
    velocities_flat = velocities.reshape((batch_size, total_nodes, 2))
    radii_flat = radii.reshape((batch_size, total_nodes, 1))

    receiver_idx, sender_idx, edge_type_values = _spatio_temporal_edge_indices(history_len, num_entities)
    receiver_pos = positions_flat[:, receiver_idx, :]
    sender_pos = positions_flat[:, sender_idx, :]
    rel_pos = sender_pos - receiver_pos
    receiver_vel = velocities_flat[:, receiver_idx, :]
    sender_vel = velocities_flat[:, sender_idx, :]
    rel_vel = sender_vel - receiver_vel
    receiver_radius = radii_flat[:, receiver_idx, :]
    sender_radius = radii_flat[:, sender_idx, :]
    clearance = jnp.linalg.norm(rel_pos, axis=-1, keepdims=True) - receiver_radius - sender_radius
    clearance = jnp.maximum(clearance, 0.0)
    edge_type = jnp.broadcast_to(edge_type_values[None, :, None].astype(dtype), (batch_size, edge_type_values.shape[0], 1))

    edges = jnp.concatenate(
        [
            rel_pos / float(hparams.position_scale),
            clearance / float(hparams.position_scale),
            rel_vel / float(hparams.velocity_scale),
            edge_type,
        ],
        axis=-1,
    )
    return nodes, edges, node_mask_flat, receiver_idx, sender_idx


def _current_outputs(nodes, hparams, return_human_latents):
    num_entities = int(hparams.num_humans) + 2
    current_offset = (int(hparams.history_len) - 1) * num_entities
    robot_latent = nodes[:, current_offset, :]
    human_latents = nodes[:, current_offset + 1 : current_offset + 1 + int(hparams.num_humans), :]
    if return_human_latents:
        return robot_latent, human_latents
    return robot_latent


def spatio_temporal_graph_encode(params, hparams, obs_history, return_human_latents=False):
    obs_history = _reshape_obs_history(obs_history, hparams.history_len, hparams.obs_frame_dim)
    nodes, edges, node_mask, receiver_idx, sender_idx = _build_spatio_temporal_graph_tensors(hparams, obs_history)
    for layer in params["layers"]:
        nodes = reference_attention_layer_sparse(layer, nodes, edges, node_mask, receiver_idx, sender_idx, act_name="relu")
    return _current_outputs(nodes, hparams, return_human_latents)


def current_obs_history(obs, hparams):
    return jnp.asarray(obs, dtype=jnp.float32).reshape(
        (obs.shape[0], hparams.history_len, hparams.obs_frame_dim)
    )


def init_shared_gnn_actor(
    keys,
    hparams,
    act_dim,
    hidden_dim,
    control_hidden_dim,
    action_std_init,
    head_name,
    use_init_weights,
):
    sqrt2 = math.sqrt(2.0)
    return {
        "logstd": jnp.full((act_dim,), math.log(float(action_std_init)), dtype=jnp.float32),
        "graph_encoder": init_spatio_temporal_encoder(
            keys[0],
            obs_frame_dim=hparams.obs_frame_dim,
            graph_hidden_dim=hparams.graph_hidden_dim,
            graph_latent_dim=hparams.graph_latent_dim,
            graph_layers=hparams.graph_layers,
            use_init_weights=use_init_weights,
        ),
        "fc1": init_linear(keys[1], hparams.graph_latent_dim, hidden_dim, sqrt2, use_init_weights),
        head_name: init_head(keys[2:4], hidden_dim, control_hidden_dim, act_dim, use_init_weights),
    }


def init_shared_gnn_critic(keys, hparams, hidden_dim, hidden_dim2, use_init_weights):
    sqrt2 = math.sqrt(2.0)
    return {
        "fc1": init_linear(keys[0], hparams.graph_latent_dim, hidden_dim, sqrt2, use_init_weights),
        "fc21": init_linear(keys[1], hidden_dim, hidden_dim2, sqrt2, use_init_weights),
        "fc31": init_linear(keys[2], hidden_dim2, 1, 1.0, use_init_weights),
    }


def actor_latent_and_mean(params, hparams, obs, head_name):
    obs_history = current_obs_history(obs, hparams)
    actor_params = params["actor"]
    robot_latent = spatio_temporal_graph_encode(actor_params["graph_encoder"], hparams, obs_history)
    hidden = activation(dense(actor_params["fc1"], robot_latent), hparams.actor_act)
    return robot_latent, mlp(actor_params[head_name], hidden, hparams.actor_act)


def critic_value_from_latent(params, hparams, robot_latent):
    critic_params = params["critic"]
    hidden = activation(dense(critic_params["fc1"], robot_latent), hparams.critic_act)
    hidden = activation(dense(critic_params["fc21"], hidden), hparams.critic_act)
    return dense(critic_params["fc31"], hidden).squeeze(-1)


def shared_critic_value(params, hparams, obs):
    obs_history = current_obs_history(obs, hparams)
    latent = spatio_temporal_graph_encode(params["actor"]["graph_encoder"], hparams, obs_history)
    return critic_value_from_latent(params, hparams, latent)
