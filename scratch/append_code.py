from typing import Iterator

def _stream_examples_iter(lang: str, limit: int, skip: int = 0, split: str = "train"):
    from datasets import load_dataset
    prefix = _LANG_TO_PREFIX.get(lang)
    if not prefix: raise ValueError(f"Unknown lang {lang}")
    
    try:
        print(f"    Trying val_parquet...")
        ds = load_dataset("parquet", data_files=f"https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/validation/{prefix}val.parquet", split="train", streaming=True)
        if skip > 0:
            ds = ds.skip(skip)
        ds = ds.take(limit)
        print(f"    [OK] val_parquet: streaming...")
        return iter(ds)
    except Exception as e:
        print(f"    [WARN] val_parquet failed: {str(e)[:100]}")
    return iter([])

def load_msmarco_xi_stream_batches(
    languages: list[str],
    sample_size: int = 10000,
    batch_size: int = 2000,
    start_offset: int = 0,
) -> Iterator[DatasetSplit]:
    load_english = "en" in languages
    indic_languages = [l for l in languages if l != "en"]
    
    english_source_lang = None
    if load_english:
        if "hi" in indic_languages: english_source_lang = "hi"
        elif indic_languages: english_source_lang = indic_languages[0]
        else:
            english_source_lang = "hi"
            indic_languages.append("hi")
            
    all_langs_to_load = list(set(indic_languages))
    
    for lang in all_langs_to_load:
        print(f"\n{'='*60}\nStreaming [{lang}] (offset={start_offset}, limit={sample_size})\n{'='*60}")
        
        limit = sample_size - start_offset
        if limit <= 0: continue
            
        iterator = _stream_examples_iter(lang, limit, skip=start_offset)
        extract_en = (load_english and lang == english_source_lang)
        
        current_batch = DatasetSplit()
        batch_count = 0
        global_idx = start_offset
        
        for example in iterator:
            passages, eval_pair, en_passages, en_eval_pair = _process_example(example, lang, global_idx, extract_english=extract_en)
            current_batch.passages.extend(passages)
            if eval_pair: current_batch.eval_pairs.append(eval_pair)
            if en_passages: current_batch.passages.extend(en_passages)
            if en_eval_pair: current_batch.eval_pairs.append(en_eval_pair)
            
            batch_count += 1
            global_idx += 1
            
            if batch_count >= batch_size:
                yield current_batch
                current_batch = DatasetSplit()
                batch_count = 0
                
        if batch_count > 0:
            yield current_batch
