import torch

from nltk.translate.bleu_score import corpus_bleu


import utils
import imp

imp.reload(utils)
generate_translation = utils.generate_translation
remove_tech_tokens = utils.remove_tech_tokens
get_text = utils.get_text
flatten = utils.flatten


def get_bleu_score(model, test_iterator, TRG):
    original_text = []
    generated_text = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_iterator):
            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0)  # turn off teacher forcing

            output = output.argmax(dim=-1)

            original_text.extend([get_text(x, TRG.vocab) for x in trg.cpu().numpy().T])
            generated_text.extend(
                [get_text(x, TRG.vocab) for x in output[1:].detach().cpu().numpy().T]
            )
    score = corpus_bleu([[text] for text in original_text], generated_text) * 100

    return score, original_text, generated_text
