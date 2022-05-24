import data_preprocessing
import get_bleu_score
import lstm_attention
import train_pipeline
import GRU_only
import utils


if __name__ == "__main__":
    (
        train_data,
        valid_data,
        test_data,
        SRC,
        TRG,
    ) = data_preprocessing.return_preprocessed_data()
    encoder = lstm_attention.Encoder
    decoder = lstm_attention.Decoder
    seq2seq = lstm_attention.Seq2Seq
    (
        baseline_model,
        train_history,
        valid_history,
        test_iterator,
    ) = train_pipeline.train_model(
        train_data,
        valid_data,
        test_data,
        SRC,
        TRG,
        GRU_only.Encoder,
        GRU_only.Decoder,
        GRU_only.Seq2Seq,
        "Baseline 20 iters",
        batch_size=32,
        n_iter=20,
    )
