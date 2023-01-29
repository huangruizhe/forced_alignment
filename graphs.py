import k2
from k2.symbol_table import SymbolTable
import torch

def make_fixed_length_ctc_topology(lexicon, num_repeats=5):
    # each frame: 0.04s
    start_state = 0
    final_state = len(lexicon.tokens) * (2 * num_repeats - 3) + 1
    arcs = []

    ins_token = lexicon.token_table.get("#INS")
    skip_token = lexicon.token_table.get("#SKIP")
    space = "|"

    eps = 0
    arcs.append([start_state, final_state, -1, -1, 0])
    arcs.append([start_state, start_state, ins_token, ins_token, -0.3])  # encourage the path to stay within the token
    arcs.append([start_state, start_state, skip_token, skip_token, -0.3])
    for token_id in lexicon.tokens:  # non-disambiguation, non-zero tokens
        if lexicon.token_table.get(token_id) == space:
            arcs.append([start_state, start_state, token_id, token_id, -0.3])
            continue

        cur_state = (token_id - 1) * (2 * num_repeats - 3) + 1
        
        token_states = [cur_state]

        arcs.append([start_state, cur_state, token_id, token_id, 0])
        arcs.append([cur_state, start_state, eps, eps, 0])

        next_state = cur_state + 1
        for _ in range(num_repeats - 2):
            arcs.append([cur_state, next_state, token_id, eps, 0])
            arcs.append([next_state, start_state, eps, eps, 0])
            token_states.append(next_state)
            cur_state = next_state
            next_state += 1
        arcs.append([cur_state, start_state, token_id, eps, 0])

        cur_state = token_states[0]
        for _ in range(num_repeats - 2):
            arcs.append([cur_state, next_state, eps, eps, 0])
            arcs.append([next_state, start_state, eps, eps, 0])
            token_states.append(next_state)
            cur_state = next_state
            next_state += 1
        
        for s1, s2 in zip(token_states[1: num_repeats - 1], token_states[num_repeats:]):
            arcs.append([s1, s2, eps, eps, 0])

        for s in token_states:
            arcs.append([s, final_state, -1, -1, 0])

        for s in token_states:
            for another_token_id in lexicon.tokens:
                if another_token_id == token_id:
                    continue
                another_state = (another_token_id - 1) * (2 * num_repeats - 3) + 1
                arcs.append([s, another_state, another_token_id, another_token_id, 0])
    
    # for s in lexicon.token_table.symbols:
    # for token in ["#INS", "#SKIP"]:
    #     token_id = lexicon.token_table.get(token)
    #     arcs.append([start_state, start_state, token_id, token_id, 0])

    assert next_state == final_state, f"next_state={next_state}, final_state={final_state}"
    arcs.append([final_state])

    arcs = sorted(arcs, key=lambda arc: arc[0])
    arcs = [[str(i) for i in arc] for arc in arcs]
    arcs = [" ".join(arc) for arc in arcs]
    arcs = "\n".join(arcs)

    fst = k2.Fsa.from_str(arcs, acceptor=False)

    # fst.labels_sym = lexicon.token_table
    return fst


def make_grammar(word_sequence, lexicon):
    eps = 0
    word2id = lexicon.word_table._sym2id
    skip = word2id["#SKIP"]
    insert = word2id["#INS"]
    space = "|"

    arcs = []
    cur_state = 0
    ins_state = 1
    ins_weight = -5.0
    skip_weight = -5.0

    assert len(word_sequence) > 0

    arcs.append([cur_state, cur_state, insert, insert, 0])  # blk at the start of the sequence

    for i, word in enumerate(word_sequence):
        next_state = cur_state + 1
        arcs.append([cur_state, next_state, word2id[word], word2id[word], 0])
        # arcs.append([cur_state, cur_state, eps, eps, 0])
        arcs.append([cur_state, next_state, skip, skip, skip_weight * len(word)])
        if i > 0:
            arcs.append([cur_state, cur_state, word2id[space], word2id[space], 0])
            arcs.append([cur_state, cur_state, insert, insert, 0])  # there can only be space between words
        cur_state += 1

    # arcs.append([cur_state, cur_state, word2id[space], word2id[space], 0])
    arcs.append([cur_state, cur_state, insert, insert, 0])  # blk at the end of the sequence
    arcs.append([cur_state, cur_state + 1, -1, -1, 0])
    arcs.append([cur_state + 1])

    arcs = sorted(arcs, key=lambda arc: arc[0])
    arcs = [[str(i) for i in arc] for arc in arcs]
    arcs = [" ".join(arc) for arc in arcs]
    arcs = "\n".join(arcs)

    fsa = k2.Fsa.from_str(arcs, acceptor=False)    

    # # label_sym_str = "\n".join([f"{word} {word_id}" for word, word_id in word2id.items()])
    # # labels_sym = k2.SymbolTable.from_str(label_sym_str)
    # fsa.labels_sym = lexicon.word_table
    # fsa.aux_labels_sym = lexicon.word_table

    return fsa


def compose_tlg(T, L, G, lexicon):
    T = k2.arc_sort(T)
    L = k2.arc_sort(L)
    G = k2.arc_sort(G)
    token2id = lexicon.token_table._sym2id
    word2id = lexicon.word_table._sym2id

    # print("Intersecting L and G")
    LG = k2.compose(L, G)
    # print(f"LG shape: {LG.shape}")

    # print("Connecting LG")
    LG = k2.connect(LG)
    # print(f"LG shape after k2.connect: {LG.shape}")

    # print("Determinizing LG")
    LG = k2.arc_sort(LG)
    LG = k2.determinize(LG)

    # print("Connecting LG after k2.determinize")
    LG = k2.connect(LG)

    LG = k2.remove_epsilon(LG)
    # print(f"LG shape after k2.remove_epsilon: {LG.shape}")

    LG = k2.arc_sort(LG)

    # print("Composing T and LG")
    # CAUTION: The name of the inner_labels is fixed
    # to `tokens`. If you want to change it, please
    # also change other places in icefall that are using
    # it.
    TLG = k2.compose(T, LG, inner_labels="tokens")

    # print("Connecting TLG")
    TLG = k2.connect(TLG)

    first_token_disambig_id = token2id["#0"]
    first_word_disambig_id = word2id["#0"]

    TLG.labels[TLG.labels >= first_token_disambig_id] = 0

    TLG.__dict__["_properties"] = None
    assert isinstance(LG.aux_labels, k2.RaggedTensor)
    TLG.aux_labels.values[TLG.aux_labels.values >= first_word_disambig_id] = 0

    # print("Arc sorting LG")
    TLG = k2.arc_sort(TLG)
    # print(f"TLG.shape: {TLG.shape} num_arcs: {TLG.num_arcs}")
    return TLG

def get_T_and_L(lexicon, lang_dir):
    T = make_fixed_length_ctc_topology(lexicon, num_repeats=5)
    L = k2.Fsa.from_dict(torch.load(f"{lang_dir}/L_disambig.pt"))
    return T, L

