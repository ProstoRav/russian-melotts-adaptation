"""Compatibility zero-BERT module for the POC.

Replace with a validated RuBERT/XLM-R token-to-phone alignment implementation
only in the quality experiment branch.
"""
import torch


def get_bert_feature(text, word2ph, device=None):
    return torch.zeros(768, sum(word2ph), device="cpu")
