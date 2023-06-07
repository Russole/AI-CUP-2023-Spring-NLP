import torch
import opencc
import numpy as np
import pandas as pd
import json

from torch import nn
from typing import Dict, List, Set, Tuple, Union
#from dataset import BERTDataset, Dataset
from dataclasses import dataclass
from hanlp.components.pipeline import Pipeline
from tqdm import tqdm

#from utils import compare_strings

@dataclass
class Claim:
    data: str

@dataclass
class AnnotationID:
    id: int

@dataclass
class EvidenceID:
    id: int

@dataclass
class PageTitle:
    title: str

@dataclass
class SentenceID:
    id: int

@dataclass
class Evidence:
    data: List[List[Tuple[AnnotationID, EvidenceID, PageTitle, SentenceID]]]


CONVERTER_T2S = opencc.OpenCC("t2s.json")
CONVERTER_S2T = opencc.OpenCC("s2t.json")

def get_pred_pages_v1(series_data: pd.Series, mapping2) -> Set[Dict[int, str]]:
    import time
    import wikipedia
    import pandas as pd
    import re
    wikipedia.set_lang("zh")
    from helper_function import do_st_corrections
    print(series_data.name)
    claim_in_candiate=[]
    nps_in_candiate=[]
    # CompareNpsCandidate=[]
    #nps_in_candiate2=[]
    results = []
    tmp_muji = []
    # wiki_page: its index showned in claim
    mapping = {}
    print(series_data["id"])
    claim = series_data["claim"]
    nps = series_data["hanlp_results"]
    first_wiki_term = [] 
    for np in nps:
      if np in mapping2:
        nps_in_candiate.append(np)
    #   for k,v in mapping2.items():
    #     count=compare_strings(np, k)
    #     if count == 1:
    #       CompareNpsCandidate.append(k)
    #print(nps_in_candiate)
    for i, np in enumerate(nps):
        # Simplified Traditional Chinese Correction
        wiki_search_results = [
            do_st_corrections(w) for w in wikipedia.search(np)
        ]

        # Remove the wiki page's description in brackets
        # wiki_set = [re.sub(r"\s\(\S+\)", "", w) for w in wiki_search_results]
        wiki_set = [w for w in wiki_search_results]
        wiki_df = pd.DataFrame({
            "wiki_set": wiki_set,
            "wiki_results": wiki_search_results
        })

        # Elements in wiki_set --> index
        # Extracting only the first element is one way to avoid extracting
        # too many of the similar wiki pages
        grouped_df = wiki_df.groupby("wiki_set", sort=False).first()
        candidates = grouped_df["wiki_results"].tolist()
        # muji refers to wiki_set
        muji = grouped_df.index.tolist()

        for candidate in candidates:
          if candidate in claim:
              #print("candidate 在claim中出現",candidate)
              claim_in_candiate.append(candidate)

        for prefix, term in zip(muji, candidates):
            if prefix not in tmp_muji:
                matched = False

                # Take at least one term from the first noun phrase
                if i == 0:
                    first_wiki_term.append(term)

                # Walrus operator :=
                # https://docs.python.org/3/whatsnew/3.8.html#assignment-expressions
                # Through these filters, we are trying to figure out if the term
                # is within the claim
                if (((new_term := term) in claim) or
                    ((new_term := term.replace("·", "")) in claim) or
                    ((new_term := term.split(" ")[0]) in claim) or
                    ((new_term := term.replace("-", " ")) in claim)):
                    matched = True

                elif "·" in term:
                    splitted = term.split("·")
                    for split in splitted:
                        if (new_term := split) in claim:
                            matched = True
                            break

                if matched:
                    # post-processing
                    term = term.replace(" ", "_")
                    term = term.replace("-", "")
                    results.append(term)
                    mapping[term] = claim.find(new_term)
                    tmp_muji.append(new_term)
        #time.sleep(0.5)

    # 5 is a hyperparameter
    if len(results) > 10:
        assert -1 not in mapping.values()
        results = sorted(mapping, key=mapping.get)[:10]
    elif len(results) < 1:
        results = first_wiki_term
    results.extend(claim_in_candiate)
    results.extend(nps_in_candiate)
    #results.extend(nps_in_candiate2)
    return set(results)

def get_pred_pages(series_data: pd.Series) -> Set[Dict[int, str]]:
    # thread import 
    import time
    import wikipedia
    import pandas as pd
    import re
    wikipedia.set_lang("zh")
    from helper_function import do_st_corrections
    # 
    results = []
    tmp_muji = []
    # wiki_page: its index showned in claim
    mapping = {}
    claim = series_data["claim"]
    nps = series_data["hanlp_results"]
    first_wiki_term = []

    for i, np in enumerate(nps):
        # Simplified Traditional Chinese Correction
        wiki_search_results = [
            do_st_corrections(w) for w in wikipedia.search(np)
        ]

        # Remove the wiki page's description in brackets
        # 
        wiki_set = [re.sub(r"\s\(\S+\)", "", w) for w in wiki_search_results]
        wiki_set = [w for w in wiki_search_results]
        wiki_df = pd.DataFrame({
            "wiki_set": wiki_set,
            "wiki_results": wiki_search_results
        })

        # Elements in wiki_set --> index
        # Extracting only the first element is one way to avoid extracting
        # too many of the similar wiki pages
        grouped_df = wiki_df.groupby("wiki_set", sort=False).first()
        candidates = grouped_df["wiki_results"].tolist()
        # muji refers to wiki_set
        muji = grouped_df.index.tolist()

        for prefix, term in zip(muji, candidates):
            if prefix not in tmp_muji:
                matched = False

                # Take at least one term from the first noun phrase
                if i == 0:
                    first_wiki_term.append(term)

                # Walrus operator :=
                # https://docs.python.org/3/whatsnew/3.8.html#assignment-expressions
                # Through these filters, we are trying to figure out if the term
                # is within the claim
                if (((new_term := term) in claim) or
                    ((new_term := term.replace("·", "")) in claim) or
                    ((new_term := term.split(" ")[0]) in claim) or
                    ((new_term := term.replace("-", " ")) in claim)):
                    matched = True

                elif "·" in term:
                    splitted = term.split("·")
                    for split in splitted:
                        if (new_term := split) in claim:
                            matched = True
                            break

                if matched:
                    # post-processing
                    term = term.replace(" ", "_")
                    term = term.replace("-", "")
                    results.append(term)
                    mapping[term] = claim.find(new_term)
                    tmp_muji.append(new_term)
        time.sleep(0.5)

    # 5 is a hyperparameter
    if len(results) > 5:
        assert -1 not in mapping.values()
        results = sorted(mapping, key=mapping.get)[:5]
    elif len(results) < 1:
        results = first_wiki_term

    return set(results)

def do_st_corrections(text: str) -> str:
    simplified = CONVERTER_T2S.convert(text)

    return CONVERTER_S2T.convert(simplified)

def get_nps_hanlp(
    predictor: Pipeline,
    d: Dict[str, Union[int, Claim, Evidence]],
) -> List[str]:
    claim = d["claim"]
    tree = predictor(claim)["con"]
    nps = [
        do_st_corrections("".join(subtree.leaves()))
        for subtree in tree.subtrees(lambda t: t.label() == "NP")
    ]

    return nps



def calculate_precision(
    data: List[Dict[str, Union[int, Claim, Evidence]]],
    predictions: pd.Series,
) -> None:
    precision = 0
    count = 0

    for i, d in enumerate(data):
        if d["label"] == "NOT ENOUGH INFO":
            continue

        # Extract all ground truth of titles of the wikipedia pages
        # evidence[2] refers to the title of the wikipedia page
        gt_pages = set([
            evidence[2]
            for evidence_set in d["evidence"]
            for evidence in evidence_set
        ])

        predicted_pages = predictions.iloc[i]
        hits = predicted_pages.intersection(gt_pages)
        if len(predicted_pages) != 0:
            precision += len(hits) / len(predicted_pages)

        count += 1

    # Macro precision
    print(f"Precision: {precision / count}")


def calculate_recall(
    data: List[Dict[str, Union[int, Claim, Evidence]]],
    predictions: pd.Series,
) -> None:
    recall = 0
    count = 0

    for i, d in enumerate(data):
        if d["label"] == "NOT ENOUGH INFO":
            continue

        gt_pages = set([
            evidence[2]
            for evidence_set in d["evidence"]
            for evidence in evidence_set
        ])
        predicted_pages = predictions.iloc[i]
        hits = predicted_pages.intersection(gt_pages)
        recall += len(hits) / len(gt_pages)
        count += 1

    print(f"Recall: {recall / count}")

def save_doc(
    data: List[Dict[str, Union[int, Claim, Evidence]]],
    predictions: pd.Series,
    mode: str = "train",
    num_pred_doc: int = 5,
) -> None:
    with open(
        f"data/Stage1/UnBM25_{mode}_doc{num_pred_doc}.jsonl",
        "w",
        encoding="utf8",
    ) as f:
        for i, d in enumerate(data):
            d["predicted_pages"] = list(predictions.iloc[i])
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

def evidence_macro_precision(
    instance: Dict,
    top_rows: pd.DataFrame,
) -> Tuple[float, float]:
    """Calculate precision for sentence retrieval
    This function is modified from fever-scorer.
    https://github.com/sheffieldnlp/fever-scorer/blob/master/src/fever/scorer.py

    Args:
        instance (dict): a row of the dev set (dev.jsonl) of test set (test.jsonl)
        top_rows (pd.DataFrame): our predictions with the top probabilities

        IMPORTANT!!!
        instance (dict) should have the key of `evidence`.
        top_rows (pd.DataFrame) should have a column `predicted_evidence`.

    Returns:
        Tuple[float, float]:
        [1]: relevant and retrieved (numerator of precision)
        [2]: retrieved (denominator of precision)
    """
    this_precision = 0.0
    this_precision_hits = 0.0

    # Return 0, 0 if label is not enough info since not enough info does not
    # contain any evidence.
    if instance["label"].upper() != "NOT ENOUGH INFO":
        # e[2] is the page title, e[3] is the sentence index
        all_evi = [[e[2], e[3]]
                   for eg in instance["evidence"]
                   for e in eg
                   if e[3] is not None]
        claim = instance["claim"]
        predicted_evidence = top_rows[top_rows["claim"] ==
                                      claim]["predicted_evidence"].tolist()

        for prediction in predicted_evidence:
            if prediction in all_evi:
                this_precision += 1.0
            this_precision_hits += 1.0

        return (this_precision /
                this_precision_hits) if this_precision_hits > 0 else 1.0, 1.0

    return 0.0, 0.0

def evidence_macro_recall(
    instance: Dict,
    top_rows: pd.DataFrame,
) -> Tuple[float, float]:
    """Calculate recall for sentence retrieval
    This function is modified from fever-scorer.
    https://github.com/sheffieldnlp/fever-scorer/blob/master/src/fever/scorer.py

    Args:
        instance (dict): a row of the dev set (dev.jsonl) of test set (test.jsonl)
        top_rows (pd.DataFrame): our predictions with the top probabilities

        IMPORTANT!!!
        instance (dict) should have the key of `evidence`.
        top_rows (pd.DataFrame) should have a column `predicted_evidence`.

    Returns:
        Tuple[float, float]:
        [1]: relevant and retrieved (numerator of recall)
        [2]: relevant (denominator of recall)
    """
    # We only want to score F1/Precision/Recall of recalled evidence for NEI claims
    if instance["label"].upper() != "NOT ENOUGH INFO":
        # If there's no evidence to predict, return 1
        if len(instance["evidence"]) == 0 or all(
            [len(eg) == 0 for eg in instance]):
            return 1.0, 1.0

        claim = instance["claim"]

        predicted_evidence = top_rows[top_rows["claim"] ==
                                      claim]["predicted_evidence"].tolist()

        for evidence_group in instance["evidence"]:
            evidence = [[e[2], e[3]] for e in evidence_group]
            if all([item in predicted_evidence for item in evidence]):
                # We only want to score complete groups of evidence. Incomplete
                # groups are worthless.
                return 1.0, 1.0
        return 0.0, 1.0
    return 0.0, 0.0

def evaluate_retrieval(
    probs: np.ndarray,
    df_evidences: pd.DataFrame,
    ground_truths: pd.DataFrame,
    top_n: int = 5,
    cal_scores: bool = True,
    save_name: str = None,
) -> Dict[str, float]:
    """Calculate the scores of sentence retrieval

    Args:
        probs (np.ndarray): probabilities of the candidate retrieved sentences
        df_evidences (pd.DataFrame): the candiate evidence sentences paired with claims
        ground_truths (pd.DataFrame): the loaded data of dev.jsonl or test.jsonl
        top_n (int, optional): the number of the retrieved sentences. Defaults to 2.

    Returns:
        Dict[str, float]: F1 score, precision, and recall
    """
    df_evidences["prob"] = probs
    top_rows = (
        df_evidences.groupby("claim").apply(
        lambda x: x.nlargest(top_n, "prob"))
        .reset_index(drop=True)
    )

    if cal_scores:
        macro_precision = 0
        macro_precision_hits = 0
        macro_recall = 0
        macro_recall_hits = 0

        for i, instance in enumerate(ground_truths):
            macro_prec = evidence_macro_precision(instance, top_rows)
            macro_precision += macro_prec[0]
            macro_precision_hits += macro_prec[1]

            macro_rec = evidence_macro_recall(instance, top_rows)
            macro_recall += macro_rec[0]
            macro_recall_hits += macro_rec[1]

        pr = (macro_precision /
              macro_precision_hits) if macro_precision_hits > 0 else 1.0
        rec = (macro_recall /
               macro_recall_hits) if macro_recall_hits > 0 else 0.0
        f1 = 2.0 * pr * rec / (pr + rec)

    if save_name is not None:
        # write doc7_sent5 file
        with open(f"data/{save_name}", "w") as f:
            for instance in ground_truths:
                claim = instance["claim"]
                predicted_evidence = top_rows[
                    top_rows["claim"] == claim]["predicted_evidence"].tolist()
                instance["predicted_evidence"] = predicted_evidence
                f.write(json.dumps(instance, ensure_ascii=False) + "\n")

    if cal_scores:
        return {"F1 score": f1, "Precision": pr, "Recall": rec}


def get_candidates(series_data: pd.Series, mapping2) -> Set[Dict[int, str]]:
    import time
    import wikipedia
    import pandas as pd
    import re
    wikipedia.set_lang("zh")
    from helper_function import do_st_corrections
    AllCandidateList=[]
    ALLgrouped_df=[]
    claim_in_candiate=[]
    nps_in_candiate=[]
    # CompareNpsCandidate=[]
    #nps_in_candiate2=[]
    results = []
    tmp_muji = []
    # wiki_page: its index showned in claim
    mapping = {}
    print(series_data["id"])
    claim = series_data["claim"]
    nps = series_data["hanlp_results"]
    first_wiki_term = [] 
    # for np in nps:
    #   if np in mapping2:
    #     nps_in_candiate.append(np)
    #   for k,v in mapping2.items():
    #     count=compare_strings(np, k)
    #     if count == 1:
    #       CompareNpsCandidate.append(k)
    #print(nps_in_candiate)
    for i, np in enumerate(nps):
        # Simplified Traditional Chinese Correction
        wiki_search_results = [
            do_st_corrections(w) for w in wikipedia.search(np)
        ]

        # Remove the wiki page's description in brackets
        # wiki_set = [re.sub(r"\s\(\S+\)", "", w) for w in wiki_search_results]
        wiki_set = [w for w in wiki_search_results]
        wiki_df = pd.DataFrame({
            "wiki_set": wiki_set,
            "wiki_results": wiki_search_results
        })

        # Elements in wiki_set --> index
        # Extracting only the first element is one way to avoid extracting
        # too many of the similar wiki pages
        grouped_df = wiki_df.groupby("wiki_set", sort=False).first()
        candidates = grouped_df["wiki_results"].tolist()
        # muji refers to wiki_set
        muji = grouped_df.index.tolist()

        # for candidate in candidates:
        #   if candidate in claim:
        #       #print("candidate 在claim中出現",candidate)
        #       claim_in_candiate.append(candidate)
        ALLgrouped_df.extend(grouped_df)
        AllCandidateList.extend(candidates)

        # for prefix, term in zip(muji, candidates):
        #     if prefix not in tmp_muji:
        #         matched = False

        #         # Take at least one term from the first noun phrase
        #         if i == 0:
        #             first_wiki_term.append(term)

        #         # Walrus operator :=
        #         # https://docs.python.org/3/whatsnew/3.8.html#assignment-expressions
        #         # Through these filters, we are trying to figure out if the term
        #         # is within the claim
        #         if (((new_term := term) in claim) or
        #             ((new_term := term.replace("·", "")) in claim) or
        #             ((new_term := term.split(" ")[0]) in claim) or
        #             ((new_term := term.replace("-", " ")) in claim)):
        #             matched = True

        #         elif "·" in term:
        #             splitted = term.split("·")
        #             for split in splitted:
        #                 if (new_term := split) in claim:
        #                     matched = True
        #                     break

        #         if matched:
        #             # post-processing
        #             term = term.replace(" ", "_")
        #             term = term.replace("-", "")
        #             results.append(term)
        #             mapping[term] = claim.find(new_term)
        #             tmp_muji.append(new_term)
        #time.sleep(0.5)

    # 5 is a hyperparameter
    # if len(results) > 10:
    #     assert -1 not in mapping.values()
    #     results = sorted(mapping, key=mapping.get)[:10]
    # elif len(results) < 1:
    #     results = first_wiki_term
    # results.extend(claim_in_candiate)
    # results.extend(nps_in_candiate)
    #results.extend(nps_in_candiate2)
    return set(AllCandidateList),set(ALLgrouped_df)
# def get_predicted_probs(
#     model: nn.Module,
#     dataloader: Dataset,
#     device: torch.device,
# ) -> np.ndarray:
#     """Inference script to get probabilites for the candidate evidence sentences

#     Args:
#         model: the one from HuggingFace Transformers
#         dataloader: devset or testset in torch dataloader

#     Returns:
#         np.ndarray: probabilites of the candidate evidence sentences
#     """
#     model.eval()
#     probs = []

#     with torch.no_grad():
#         for batch in tqdm(dataloader):
#             batch = {k: v.to(device) for k, v in batch.items()}
#             outputs = model(**batch)
#             logits = outputs.logits
#             probs.extend(torch.softmax(logits, dim=1)[:, 1].tolist())

#     return np.array(probs)