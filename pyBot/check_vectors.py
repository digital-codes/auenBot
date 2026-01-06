import json
import numpy as np 

with open("intent_vectors.json") as f:
    data = json.load(f)
    
intent_vectors = data["vectors"]
intent_ids = data["intents"]

# convert each vector to numpy float32 and validate size
processed = []
for i, v in enumerate(intent_vectors):
    arr = np.asarray(v, dtype=np.float32)
    if arr.size != 1024:
        raise ValueError(f"Intent vector at index {i} (id={intent_ids[i] if i < len(intent_ids) else 'unknown'}) has size {arr.size}, expected 1024")
    norm = np.linalg.norm(arr)
    if norm == 0:
        raise ValueError(f"Intent vector at index {i} (id={intent_ids[i] if i < len(intent_ids) else 'unknown'}) has zero norm and cannot be normalized")
    arr = (arr / np.float32(norm)).astype(np.float32)
    processed.append(arr)

# replace intent_vectors with the processed list and also provide a stacked matrix
intent_vectors = processed
intent_matrix = np.stack(processed, axis=0)  # shape: (len(intent_vectors), 1024)

# optional mapping from intent id -> vector
intent_map = dict(zip(intent_ids, intent_vectors))

ids_arr = np.array(intent_ids)
N = intent_matrix.shape[0]

results = []

for i in range(N):
    sims = intent_matrix.dot(intent_matrix[i])  # cosine similarities to all vectors
    group = intent_ids[i]

    same_idx = np.where(ids_arr == group)[0]
    sims_same = sims[same_idx]
    max_sim = float(np.max(sims_same))
    min_sim = float(np.min(sims_same))
    max_dist = float(1.0 - min_sim)
    # second-best similarity within the same group (exclude the vector itself)
    same_other_idx = same_idx[same_idx != i]
    if same_other_idx.size == 0:
        second_best_similarity = None
    else:
        second_best_similarity = float(np.max(sims[same_other_idx]))

    other_idx = np.where(ids_arr != group)[0]
    top_list = []
    if other_idx.size > 0:
        sims_other = sims[other_idx]
        k = min(5, sims_other.size)
        top_rel = np.argsort(-sims_other)[:k]  # indices into sims_other sorted by descending similarity
        top_idx = other_idx[top_rel]
        for idx in top_idx:
            top_list.append({
                "index": int(idx),
                "intent_id": intent_ids[int(idx)],
                "similarity": float(sims[int(idx)]),
                "distance": float(1.0 - sims[int(idx)])
            })

    results.append({
        "index": int(i),
        "intent_id": group,
        "own_group_max_similarity": max_sim,
        "own_group_min_similarity": min_sim,
        "own_group_max_distance": max_dist,
        "own_group_second_best_similarity": second_best_similarity,
        "own_group_second_best_index": int(same_other_idx[np.argmax(sims[same_other_idx])]) if same_other_idx.size > 0 else None,
        "top_5_similar": top_list
    })

with open("vector_checks.json", "w") as f:
    json.dump(results, f, indent=2)