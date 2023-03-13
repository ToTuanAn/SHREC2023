import torch
from src.utils.retriever import FaissRetrieval
import tqdm

def inference(config):
    test_transforms = transforms.Compose([ Normalize(),
                                            ToTensor()])

    test_set = InferenceDataset(root_dir=cfg["dataset"]["val"], pc_transform=test_transforms, stage='train')
    test_loader = DataLoader(dataset=test_set, batch_size=1)

    model = MODEL_REGISTRY.get(cfg["model"]["name"])(cfg)
    model = model.load_from_checkpoint(pretrained_ckpt, cfg=cfg, strict=True)
    model.eval()
    gallery_embeddings = []
    query_embeddings = []
    retriever = FaissRetrieval(dimension=dimension, cpu=True) # Temporarily use CPU to retrieve (avoid OOM)
    
    obj_embedder.eval()
    query_embedder.eval()
    query_ids = []
    target_ids = []
    gallery_ids = []
    print('- Evaluation started...')
    print('- Extracting embeddings...')
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dl), total=len(dl)):
            g_emb = obj_embedder(batch[obj_input].to(device))
            q_emb = query_embedder(batch[query_input].to(device))
            gallery_embeddings.append(g_emb.detach().cpu().numpy())
            query_embeddings.append(q_emb.detach().cpu().numpy())
            query_ids.extend(batch['query_ids'])
            gallery_ids.extend(batch['gallery_ids'])
            target_ids.extend(batch['gallery_ids'])

    max_k = len(gallery_ids) # retrieve all available gallery items
    query_embeddings = np.concatenate(query_embeddings, axis=0)
    gallery_embeddings = np.concatenate(gallery_embeddings, axis=0)
    print('- Calculating similarity...')
    top_k_scores_all, top_k_indexes_all = retriever.similarity_search(
            query_embeddings=query_embeddings,
            gallery_embeddings=gallery_embeddings,
            top_k=max_k,
            query_ids=query_ids, target_ids=target_ids, gallery_ids=gallery_ids,
            save_results="temps/query_results.json"
        )

    model_labels = np.array(encode_labels(query_ids), dtype=np.int32)
    rank_matrix = top_k_indexes_all

    np.savetxt("temps/rank_matrix.csv", rank_matrix, delimiter=",",fmt='%i')
    print('- Evaluation results:')
    print_results(evaluate(rank_matrix, model_labels, model_labels))
