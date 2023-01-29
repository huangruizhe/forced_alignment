import logging
import argparse
import re
import torch
from tqdm import tqdm
from lhotse import CutSet
from torch.utils.data import DataLoader
import k2
from lhotse.dataset import (
    DynamicBucketingSampler,
    K2SpeechRecognitionDataset,
    PrecomputedFeatures,
    SimpleCutSampler,
    AudioSamples,
)
from graphs import get_T_and_L, make_grammar, compose_tlg
from utils.lexicon import Lexicon
from utils.text import clean_transcript
from utils.model import load_model

# from lhotse.dataset.input_strategies import OnTheFlyFeatures
# from lhotse import load_manifest, Fbank, FbankConfig, CutSet


logging.basicConfig(
    format = "%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
    level = 10
)

def parse_opts():
    parser = argparse.ArgumentParser(
        description='',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--cuts', type=str, default=None, help='')
    parser.add_argument('--exp-dir', type=str, help='')
    parser.add_argument('--lang-dir', type=str, help='')

    opts = parser.parse_args()
    logging.info(f"Parameters: {vars(opts)}")
    return opts


def get_lattice(
    nnet_output: torch.Tensor,
    decoding_graph: k2.Fsa,
    supervision_segments: torch.Tensor,
    search_beam: float,
    output_beam: float,
    min_active_states: int,
    max_active_states: int,
    subsampling_factor: int = 1,
) -> k2.Fsa:
    """Get the decoding lattice from a decoding graph and neural
    network output.
    Args:
      nnet_output:
        It is the output of a neural model of shape `(N, T, C)`.
      decoding_graph:
        An Fsa, the decoding graph. It can be either an HLG
        (see `compile_HLG.py`) or an H (see `k2.ctc_topo`).
      supervision_segments:
        A 2-D **CPU** tensor of dtype `torch.int32` with 3 columns.
        Each row contains information for a supervision segment. Column 0
        is the `sequence_index` indicating which sequence this segment
        comes from; column 1 specifies the `start_frame` of this segment
        within the sequence; column 2 contains the `duration` of this
        segment.
      search_beam:
        Decoding beam, e.g. 20.  Smaller is faster, larger is more exact
        (less pruning). This is the default value; it may be modified by
        `min_active_states` and `max_active_states`.
      output_beam:
         Beam to prune output, similar to lattice-beam in Kaldi.  Relative
         to best path of output.
      min_active_states:
        Minimum number of FSA states that are allowed to be active on any given
        frame for any given intersection/composition task. This is advisory,
        in that it will try not to have fewer than this number active.
        Set it to zero if there is no constraint.
      max_active_states:
        Maximum number of FSA states that are allowed to be active on any given
        frame for any given intersection/composition task. This is advisory,
        in that it will try not to exceed that but may not always succeed.
        You can use a very large number if no constraint is needed.
      subsampling_factor:
        The subsampling factor of the model.
    Returns:
      An FsaVec containing the decoding result. It has axes [utt][state][arc].
    """
    dense_fsa_vec = k2.DenseFsaVec(
        nnet_output,
        supervision_segments,
        allow_truncate=subsampling_factor - 1,
    )

    lattice = k2.intersect_dense_pruned(
        decoding_graph,
        dense_fsa_vec,
        search_beam=search_beam,
        output_beam=output_beam,
        min_active_states=min_active_states,
        max_active_states=max_active_states,
    )

    return lattice

def one_best_decoding(
    lattice: k2.Fsa,
    use_double_scores: bool = True,
    lm_scale_list = None,
):
    """Get the best path from a lattice.
    Args:
      lattice:
        The decoding lattice returned by :func:`get_lattice`.
      use_double_scores:
        True to use double precision floating point in the computation.
        False to use single precision.
      lm_scale_list:
        A list of floats representing LM score scales.
    Return:
      An FsaVec containing linear paths.
    """
    if lm_scale_list is not None:
        ans = dict()
        saved_am_scores = lattice.scores - lattice.lm_scores
        for lm_scale in lm_scale_list:
            am_scores = saved_am_scores / lm_scale
            lattice.scores = am_scores + lattice.lm_scores

            best_path = k2.shortest_path(lattice, use_double_scores=use_double_scores)
            key = f"lm_scale_{lm_scale}"
            ans[key] = best_path
        return ans

    return k2.shortest_path(lattice, use_double_scores=use_double_scores)


def main(opts):
    model, labels, sample_rate, device = load_model()
    space_token = "|"

    cuts = CutSet.from_file(opts.cuts)
    cuts = cuts.resample(sample_rate)
    cuts.describe()

    lexicon = Lexicon(opts.lang_dir)
    lexicon.disambig_pattern = re.compile(r"^#.+$")

    T, L = get_T_and_L(lexicon, opts.lang_dir)

    # for cut in tqdm(cuts):
    #     assert len(cut.supervisions) == 1
    #     supervision = cut.supervisions[0]
    #     uid = supervision.id
    #     transcript = supervision.text
    #     audio = cut.load_audio()
    #     # audiofile = cut.recording.sources[0].source
    #     # offset = cut.start + supervision.start
    #     # duration = supervision.duration

    #     with torch.inference_mode():
    #         emissions, _ = model(audio.to(device))
    #         emissions = torch.log_softmax(emissions, dim=-1)

    # https://lhotse.readthedocs.io/en/latest/datasets.html#lhotse.dataset.speech_recognition.K2SpeechRecognitionDataset
    dataset = K2SpeechRecognitionDataset(
        input_strategy=AudioSamples(),
        # input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80))),
        return_cuts=True,
    )
    sampler = DynamicBucketingSampler(cuts, max_duration=200, num_buckets=min(30, len(cuts)))
    dl = DataLoader(dataset, batch_size=None, sampler=sampler, num_workers=0)

    num_cuts = 0
    log_interval = 20
    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"

    for batch_idx, batch in enumerate(dl):
        audios = batch['inputs']
        supervisions = batch["supervisions"]
        texts = batch["supervisions"]["text"]
        texts = [clean_transcript(text, labels, upper=True, space_symbol=space_token) for text in texts]
        cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]

        Gs = []
        for text in texts:
            Gs.append(make_grammar(text.split(space_token), lexicon))
        Gs = k2.create_fsa_vec(Gs)
        TLGs = compose_tlg(T, L, Gs, lexicon)
        TLGs = TLGs.to(device)
        assert TLGs.requires_grad is False

        with torch.inference_mode():
            emissions, emissions_lengths = model(
                audios.to(device), 
                lengths=supervisions["num_samples"].to(device)
            )
            emissions = torch.log_softmax(emissions, dim=-1)
        nnet_output = emissions
        
        supervision_segments = torch.stack(
            (
                supervisions["sequence_idx"],  # Column 0 is the sequence_index indicating which sequence this segment comes from;
                supervisions["start_sample"],  # column 1 specifies the start_frame of this segment within the sequence;
                emissions_lengths.cpu(), # column 2 contains the duration of this segment.
            ),
            1,
        ).to(torch.int32)

        lattice = get_lattice(
            nnet_output=nnet_output,
            decoding_graph=TLGs,
            supervision_segments=supervision_segments,
            search_beam=40,
            output_beam=12,
            min_active_states=15000,
            max_active_states=100000,
            subsampling_factor=1,
        )
        lattice = k2.connect(lattice)

        print(lattice.shape, lattice.num_arcs)

        # best_path = one_best_decoding(
        #     lattice=lattice, use_double_scores=False
        # )
        
        num_cuts += len(texts)

        if batch_idx % log_interval == 0:
            batch_str = f"{batch_idx}/{num_batches}"
            logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")


if __name__ == '__main__':
    opts = parse_opts()

    main(opts)
