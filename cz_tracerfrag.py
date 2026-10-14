import os, sys
import numpy as np

import libmf.micflash as mflash
import libmf.tools as tools
import libmf.micplot as mplot
from libmf.tools import fprint

import cz_tools as cz

import cl_find_181 as clfind
import cz_tracerown as cztrown
import cz_tracermatch as cztrmatch

from enum import IntEnum


#===============================================================================
# ==== GENERAL HELPER FUNCTIONS ================================================
#===============================================================================

def FilterBestFrac(I, fractions, eps=0.05):
    bestval = np.max([fractions[i] for i in I])
    I_filtered = [i for i in I if fractions[i] >= (bestval-eps)]
    return I_filtered


#===============================================================================
# ==== OBJECT RELATION HELPER FUNCTIONS ========================================
#===============================================================================

def IsAncestorLR(i1,i2, objmasks):
    ''' Check if objects i1 or i2 are ancestor of the other '''
    # This would mean that i1 fully contains i2 or vice versa
    M1 = objmasks[i1]
    M2 = objmasks[i2]
    #assert not np.all(M1==M2)
    assert np.any(M1)
    assert np.any(M2)
    is_ancestor_LR = np.all(M1*M2==M1) or np.all(M1*M2==M2)
    return is_ancestor_LR

def GroupByLinage(Iobj, objmasks):
    """
    Group predecessor nodes that are ancestor/descendant related
    in the dendrogram of snapshot A.
    """
    # XXX This is a very weird way to slice up a tree structure / multiple tree structures.
    # XXX I feel this is should not work but the logic seems to work out.
    groups = []
    for i in Iobj:
        for g in groups:
            # same lineage iff node is ancestor/descendant of everything already in group
            if np.all([IsAncestorLR(i,k, objmasks) for k in g]):
                g.append(i)
                break
        else: # loop did not break, i.e. i could not be placed in an existing group
            groups.append([i,]) # put i in new group
    assert np.sum([len(g) for g in groups]) == len(Iobj)
    return groups

def NormalizeWeights(weights_raw, mask=None):
    if mask is None: mask = np.ones(weights_raw.shape, dtype=bool)
    if not np.any(mask): return weights_raw
    weights_normalized = (weights_raw*mask)/np.sum(weights_raw*mask)
    return weights_normalized

#===============================================================================
# ==== OBJECT RELATION HELPER FUNCTIONS ========================================
#===============================================================================

def ChooseBestPredecessorLexi(j, i_group, fracmatrices, nmatrices, depth_L):

    """
    Choose the best representative from one ancestor/descendant-related
    predecessor group in snapshot A for explaining Bj in snapshot B.
    Lexicographic version.
    """
    Fcc, Rcc, Jcc, Fco, Rco, Jco, Foc, Roc, Joc, Foo, Roo, Joo = fracmatrices
    i_f1 = FilterBestFrac(i_group, Joo[:,j]) # Choose best identity-exact predecessors
    i_f2 = FilterBestFrac(i_f1, Roc[:,j])
    i_f3 = FilterBestFrac(i_f2, Rcc[:,j])
    best_i = max(i_f3, key=(lambda i: depth_L[i].item()))
    return best_i

def ChooseBestSuccessorLexi(i, j_group, fracmatrices, nmatrices, depth_R):
    """
    Choose the best representative from one ancestor/descendant-related
    successor group in snapshot B for continuing Ai from snapshot A.
    Lexicographic version.
    """
    Fcc, Rcc, Jcc, Fco, Rco, Jco, Foc, Roc, Joc, Foo, Roo, Joo = fracmatrices
    j_f1 = FilterBestFrac(j_group, Joo[i,:]) # Choose best identity-exact successors
    j_f2 = FilterBestFrac(j_f1, Foc[i,:])
    j_f3 = FilterBestFrac(j_f2, Fcc[i,:])
    best_j = max(j_f3, key=(lambda j: depth_R[j].item()))
    return best_j


#===============================================================================
# ==== OBJECT RELATION TAXONOMY ================================================
#===============================================================================

class NodeType(IntEnum):
    UNASSIGNED = 999

    IN_CONTINUED = 101
    IN_CONTINUED_ACCRETION = 111
    IN_CONTINUED_CAPTURE = 121
    IN_CONTINUATION_MERGER = 131
    IN_CONTINUATION_AMBIGUOUS_WEAK = 181
    IN_CONTINUATION_AMBIGUOUS_MULTI = 191
    IN_BORN_NOISY = 391
    IN_BORN_DIFFUSE = 381
    IN_MERGE_BINARY = 221
    IN_MERGE_MULTI = 231
    IN_BORN_INPARENT = 311
    IN_RESEGMENTATION = 331
    IN_BORN_CREATED = 301

    OUT_CONTINUED = 102
    OUT_CONTINUED_DRAIN = 112
    OUT_CONTINUED_SHEDDING = 122
    OUT_CONTINUATION_SPLIT = 132
    OUT_CONTINUATION_AMBIGUOUS_WEAK = 182
    OUT_CONTINUATION_AMBIGUOUS_MULTI = 192
    OUT_LOST_VANISHED = 392
    OUT_LOST_DISPERSED = 382
    OUT_SPLIT_BINARY = 222
    OUT_SPLIT_MULTI = 232
    OUT_ABSORP_INPARENT = 312
    OUT_RESEGMENTATION = 332
    OUT_LOST_DESTROYED = 302


class EdgeType(IntEnum):
    # Lower value <~=> Higher priority

    # Put same label on all continuations
    CONTINUED_MAIN = 1
    IN_CONTINUED_MAIN = 1
    OUT_CONTINUED_MAIN = 1
    # Put same label on all resegmentations
    RESEGMENTATION = 2
    IN_RESEGMENTATION = 2
    OUT_RESEGMENTATION = 2
    # Combined processes
    TRANSFER = 3
    FRAGMENTATION = 4
    ABSORPTION = 5
    CONTINUED_AMBIGUOUS = 6
    # Reposessions can be a mass transfers between objects of starkly different size
    # or dendrogram-related (non-physical) resegmentations
    ASYMMETRIC_REPOSESSION = 7
    #
    RESEGMENTATION_MAIN = 8
    #...
    NOISE = 99

    IN_CONT_MERGE1 = 201
    IN_MERGE1 = 211
    IN_CONT_AMBIGUOUS_BEST = 301
    IN_PARENT_BIRTH = 401
    IN_CONT_CAPTURE = 521
    IN_CONT_MERGE2 = 611
    IN_MERGE2 = 621
    IN_CONT_ACCRETION = 631
    IN_CONT_AMBIGUOUS_ALT = 701
    IN_WEAK_PREDECESSORS = 801

    OUT_CONT_SPLIT1 = 202
    OUT_SPLIT1 = 312
    OUT_CONT_AMBIGUOUS_BEST = 302
    OUT_CHILD_DEATH = 402
    OUT_CONT_SHEDDING = 522
    OUT_CONT_SPLIT2 = 612
    OUT_SPLIT2 = 622
    OUT_CONT_DRAIN = 632
    OUT_CONT_AMBIGUOUS_ALT = 702
    OUT_WEAK_SUCCESSORS = 802

    UNASSIGNED = 999

    # These will be used later for in-dendrogram relationship
    PARENT = 1001
    CHILD = 1002
    #ROOT = 1003
    #LEAF = 1004
    ANCESTOR = 1005
    DESCENDANT = 1006

    # These will be later used to help traversal
    CONT_FIRST = 1101

#===============================================================================
# ==== OBJECT RELATION THRESHOLDS ==============================================
#===============================================================================

# --- Threshold values
# Idendity candidates
Noo_min = 3
TJoo_id, TFoo_id, TRoo_id = .25, .4, .4
# Flow candidates
Ncc_min = 5
TJcc_flow, TFcc_flow, TRcc_flow = .008, .015, .015#.08, .15, .15
# Identity 1-to-1 matching
TFcc_1to1, TRcc_1to1 = .85, .85
T_dom_out, T_dom_in = .7, .7
# Identity 1-to-1 with accretion/outflow
T_minor_acc, T_minor_drain = 1e-4, 1e-4
# Splitting/merging with or without dominant branch?
T_dom_merge, T_dom_split = .7, .7 # Threshold: Dominant branch present? 
T_2_merge, T_2_split = .1, .1
# Noisy appearance / vanish
T_tiny_max, T_vanish_max = .01, .01 # upper limit
# Many weak predecessors
H_high = 0.75
# Emergence/absorption vs parent
Tnc_parent_factor = 3.
Tco_born_parent, Tco_absorp_parent = .98, .98 # XXX VERY UNSURE
Toc_born_parent, Toc_absorp_parent = 1e-5, 1e-5 # XXX VERY UNSURE
Too_cont_max = .7 # XXX VERY UNSURE: Why Limit the ownership transfer? To avoid pseudo-ID confusion?


#===============================================================================
# ==== REVERSE OBJECT RELATION GRAPH ===+=======================================
#===============================================================================

def RelationGraphR(fracmatrices, trcounts, objmasks_L, debug=True):
    ''' 
    Build a relation graph: Sort edges connecting any dendrogram node A_i
    of the previous step to each dendrogram node B_i in the current step. 
    '''
    # ------------------------------------------------------
    # Unpack tracer overlap matrices
    # ------------------------------------------------------
    # Explanation:
    # * c and o subscripts correspond to contained and owned tracers.
    #   A tracer is owned by a dendrogram node if it is directly contained,
    #   i.e. there is no contained subnode that in turn contains the tracer.
    # * Ntracer are the tracer counts for each Ai or Bj, respectively.
    # * F,R and C are forward, reverse and jaccard fractions, i.e.
    #   which fraction of Ai is found in Bj (Forward),
    #   which fraction of Bj is found in Ai (Reverse),
    #   and to which fraction do  Ai and Bj agree (cut/union) (Jaccard).
    # * N are the absolute (non-fractional) tracer counts for the corresponding (Ai,Bj) tracer set unions.
    Ntracer_Ac, Ntracer_Bc, Ntracer_Ao, Ntracer_Bo = trcounts
    Fcc, Rcc, Jcc, Fco, Rco, Jco, Foc, Roc, Joc, Foo, Roo, Joo = fracmatrices
    nmatrices = OverlapCountMatrices(fracmatrices, trcounts)
    Ncc, Nco, Noc, Noo = nmatrices
    nodedepth_L = cztrown.GetNodeDepth(objmasks_L)
    
    # ------------------------------------------------------
    # --- Prepare result fields
    # ------------------------------------------------------
    # Edge label matrix expressing type of each possible
    # edge from Ai to Bj in form of an integer enumerating the edge type.
    # We will initialize with 0 (not assigned).
    dshape = Fcc.shape
    nA, nB = dshape
    edgelabels_Bin = np.full(dshape, EdgeType.UNASSIGNED, dtype=np.int32)
    nodelabels_Bin = np.full((nB,), NodeType.UNASSIGNED, dtype=np.int32)

    # --- Check that input data contains enough nodes to work on.
    #     The following code is hopeless and will go wrong if not...
    try:
        assert nA>2
        assert nB>2
    except AssertionError:
        print(f'WARNING: RelationGraphR called for {nA}x{nB} (<3) nodes.')
        return nodelabels_Bin, edgelabels_Bin

    # ------------------------------------------------------
    # --- Find relevant connections:
    # ------------------------------------------------------
    #     Build sparse edge set using shared-tracer minimum and fraction thresholds.
    # Identity edge candidates: Ai and Bj might be the same node
    id_candidates = (Noo>Noo_min)*((Joo>TJoo_id)|(Foo>TFoo_id)|(Roo>TRoo_id))
    # Material flow edge candidate
    flow_candidates = (Ncc>Ncc_min)*((Jcc>TJcc_flow)|(Fcc>TFcc_flow)|(Rcc>TRcc_flow))
    
    # --- Primary sucessor / predecessor:
    #     Find mutual-best owned-owned matches as primary continuations
    # Primary identity successors/predecessors
    s_id_A = np.argmax(Joo, axis=1) # Which Bj is most probable identity sucessor of Ai? (Index in B)
    p_id_B = np.argmax(Joo, axis=0) # Which Ai is most probable identity predecessor of Bj? (Index in A)
    # Mutual-best pair matches:
    # Where does s_id_A match up with p_id_B, i.e. s_id_A[i]==j and p_id_B[j]==i ?
    i_,_j = np.meshgrid(np.arange(nA),np.arange(nB),indexing='ij')
    id_pairs = (s_id_A[:,None]==_j)*(p_id_B[None,:]==i_) # XXX THIS MIGHT BE WRONG XXX
    id_best = id_candidates * id_pairs

    # --- Id flow diagnostics
    # This will later be used to verify 1-to-1 object identities.
    # Note that we normalize under the identity mask to disqualify multiple strong Fcc or Rcc matches.
    # Normalize outgoing flow weights Fcc
    S_out = np.sum(Fcc*id_candidates, axis=1)+1e-10
    Ps_out = Fcc*id_candidates / S_out[:,None]
    # Normalize incoming flow weights Rcc
    S_in = np.sum(Rcc*id_candidates, axis=0)+1e-10
    Pp_in = Rcc*id_candidates / S_in[None,:]

    # ------------------------------------------------------
    # --- Decision tree (Nodes Bj, incoming)
    # ------------------------------------------------------    
    for j in range(nB):
        # * Step 0: Compute hierarchy-selective predecessors
        #   This step is necessary to eg. avoid wrong mergers of two nodes that are in a linage relationship
        # * A. GET RAW PREDECESSOR CANDIDATES
        M_flow_predecessors_raw = flow_candidates[:,j]
        I_flow_predecessors_raw = np.where(M_flow_predecessors_raw)[0]
        degree_in_raw = np.sum(M_flow_predecessors_raw)
        # * B. DO HIERARCHY PRUNING OF FLOW PREDECESSOR CANDIDATES
        #   Idea: 1. Slice predecessor candidates into groups connected through linage
        #            (ie. each pair of group members is in a ancestry (subnode) relationship).
        #         2. From each linage group choose the best possible predecessor.
        if degree_in_raw > 0:
            linage_groups = GroupByLinage(I_flow_predecessors_raw, objmasks_L)
            I_predecessors = [ChooseBestPredecessorLexi(j, Ig, fracmatrices, nmatrices, nodedepth_L) for Ig in linage_groups]
            assert len(I_predecessors) == len(set(I_predecessors)) # Sanity check: No double i from algo
        else:
            I_predecessors = list()
        M_predecessors = np.zeros(nA, dtype=bool)
        M_predecessors[I_predecessors] = True
        degree_in = len(I_predecessors)
        # Sanity checks
        assert len(I_predecessors) == np.sum(M_predecessors)
        assert np.all(M_predecessors*flow_candidates[:,j] == M_predecessors)
        if degree_in_raw > 0: assert degree_in > 0
        # * C. CALCULATE DIAGNOSTICS/WEIGHTS BEFORE AND AFTER PRUNING
        #     This will later tell continuation vs merge
        # Predecessor weights
        pred_weights_in = NormalizeWeights(Rcc[:,j], M_predecessors)
        pred1_in, pred2_in = np.sort(pred_weights_in)[[-1,-2]]
        if not pred2_in < .5:
            print(f'WARNING: pred1_in={pred1_in}, pred2_in={pred2_in}>=0.5')
        # Index of best predecessor (will be random if there is none)
        iw1_in = np.argmax(pred_weights_in)
        # Incoming normalized entropy (Note that this will sum to 0 for degree==0):
        Pf_in = pred_weights_in[M_predecessors]
        assert np.all(Pf_in>0.)
        H_in = -np.sum(Pf_in*np.log(Pf_in)) / np.log(max(2,degree_in))

        # ------------------------------------------------------    
        # * Path 1: Bj has a strong mutual-best identity predecessor
        # ------------------------------------------------------    
        if np.any(id_best[:,j]):
            assert np.sum(id_best[:,j]) == 1
            i = np.argmax(id_best[:,j])
            # * Step 2: Is the material flow (i.e. contained material) also mostly identical?
            one_to_one_match  = (Fcc[i,j]>TFcc_1to1) 
            one_to_one_match *= (Rcc[i,j]>TRcc_1to1)
            one_to_one_match *= (Ps_out[i,j]>T_dom_out) # XXX Really necessary?
            one_to_one_match *= (Pp_in[i,j]>T_dom_in) # XXX Really necessary?
            if one_to_one_match:
                wi = np.where((Roo)[:,j]>T_minor_acc)[0]
                wii = wi[wi!=i] # keep all but i itself
                extra_mass_source = len(wii)>0 # == 1 would cover the match itself
                if extra_mass_source:
                    nodelabels_Bin[j] = NodeType.IN_CONTINUED_ACCRETION
                    edgelabels_Bin[wii,j] = EdgeType.IN_CONT_ACCRETION
                    edgelabels_Bin[i,j] = EdgeType.IN_CONTINUED_MAIN
                else:
                    nodelabels_Bin[j] = NodeType.IN_CONTINUED
                    edgelabels_Bin[i,j] = EdgeType.IN_CONTINUED_MAIN
            else:
                capture = (degree_in>=2) * (pred1_in>T_dom_merge) * (pred2_in>T_2_merge)
                merger  = (degree_in>=2) * (pred1_in<=T_dom_merge) * (pred2_in>T_2_merge)
                if debug > 0:
                    pass
                    #print(f'M/C: (degree_in:{degree_in}>=2) * (pred1_in:{pred1_in}<>{T_dom_merge}) * (pred2_in:{pred2_in}>{T_2_merge})')
                # Note that capture does not require degree>2, and w2_in might be a value from a pruned edge
                if capture:
                    # Merger onto one main persistent node
                    nodelabels_Bin[j] = NodeType.IN_CONTINUED_CAPTURE
                    # ORDER MATTERS: PRIORITY OVERWRITE: ACCRETION < CAPTURE < MAIN ROUTE
                    wi = np.where(pred_weights_in>T_minor_acc)[0]
                    edgelabels_Bin[wi,j] = EdgeType.IN_CONT_ACCRETION
                    wi = np.where(pred_weights_in>T_2_merge)[0]
                    edgelabels_Bin[wi,j] = EdgeType.IN_CONT_CAPTURE
                    edgelabels_Bin[i,j] = EdgeType.IN_CONTINUED_MAIN
                elif merger:
                    # * Step 4: Merger without dominant flow but with several weak predecessors
                    nodelabels_Bin[j] = NodeType.IN_CONTINUATION_MERGER
                    wi = np.where(pred_weights_in>T_2_merge)[0]
                    edgelabels_Bin[wi,j] = EdgeType.IN_CONT_MERGE2
                    edgelabels_Bin[i,j] = EdgeType.IN_CONT_MERGE1
                else:
                    wi = np.where(id_candidates[:,j])[0]
                    wii = wi[wi!=i]
                    if debug > 0:
                        print('IN_CONTINUATION_AMBIGUOUS:')
                        print(f'(Fcc[{i},{j}]>TFcc_1to1): {(Fcc[i,j]>TFcc_1to1)} # {Fcc[i,j]} > {TFcc_1to1} ?')
                        print(f'(Rcc[{i},{j}]>TRcc_1to1): {(Rcc[i,j]>TRcc_1to1)} # {Rcc[i,j]} > {TRcc_1to1} ?')
                        print(f'(Ps_out[{i},{j}]>T_dom_out): {(Ps_out[i,j]>T_dom_out)} # {Ps_out[i,j]} > {T_dom_out} ?')
                        print(f'(Pp_in[{i},{j}]>T_dom_in): {(Pp_in[i,j]>T_dom_in)} # {Pp_in[i,j]} > {T_dom_in} ?')
                        print(f'Best candidate: A{i}')
                        print(f'Alternative candidates: A{wii}')
                    if len(wii) > 0:
                        nodelabels_Bin[j] = NodeType.IN_CONTINUATION_AMBIGUOUS_MULTI
                    else:
                        nodelabels_Bin[j] = NodeType.IN_CONTINUATION_AMBIGUOUS_WEAK
                    edgelabels_Bin[wii,j] = EdgeType.IN_CONT_AMBIGUOUS_ALT
                    edgelabels_Bin[i,j] = EdgeType.IN_CONT_AMBIGUOUS_BEST
                        
        # ------------------------------------------------------    
        # * Path 2: Bj has no mutual best predecessor
        # ------------------------------------------------------    
        else:
            # XXX MAYBE NEED TO CHANGE LOGIC:
            # * NEED BEST FLOW EQUIV. (CC-SUCCESSOR) FOR MERGERS
            # * NEED BEST CONTAINER (MIXED-SUCCESSOR (WHICH ONE? CO?) IN_PARENT STUFF
            incoming_flow_tiny = (np.sum(Rcc[:,j])<=T_tiny_max)
            if incoming_flow_tiny:
                # No strong continuation and no significant predecessor flow
                nodelabels_Bin[j] = NodeType.IN_BORN_NOISY
            else:
                many_weak_predecessors = (degree_in>2)*(H_in>H_high)
                has_any_predecessor = (degree_in>0)
                if many_weak_predecessors:
                    nodelabels_Bin[j] = NodeType.IN_BORN_DIFFUSE
                    edgelabels_Bin[I_predecessors,j] = EdgeType.IN_WEAK_PREDECESSORS
                elif has_any_predecessor:
                    # Strong material flow from one node Ai
                    i = iw1_in  # Index of best flow predecessor of Bj (i.e. most similar by containment).
                    # * Step 6: True merger (i.e. several non-dominant but substantial predecessors)?
                    ### Note: The (pred1_in<=T_dom_merge) condition distinguishes true merging from
                    #         mass transfer by capture (resegmentation)
                    true_merger = (degree_in>=2) * (pred1_in<=T_dom_merge) * (pred2_in>T_2_merge)
                    if true_merger:
                        wi = np.where(pred_weights_in>T_2_merge)[0]
                        edgelabels_Bin[wi,j] = EdgeType.IN_MERGE2
                        edgelabels_Bin[i,j] = EdgeType.IN_MERGE1
                        if (degree_in==2):
                            nodelabels_Bin[j] = NodeType.IN_MERGE_BINARY
                        elif (degree_in>=3):
                            nodelabels_Bin[j] = NodeType.IN_MERGE_MULTI
                    else: # (not a true merger)
                        # * Step 7: Newly emerged child inside old parent? # XXX THIS IS CRUCIAL TO GET RIGHT
                        # XXX BUG ? WHY IS THIS IN THE has_any_predecessor SECTION? THIS IS PROBABLY WRONG!
                        born_inside_parent  = (Fco[i,j]>Tco_born_parent) | (Roc[i,j]>Toc_born_parent)
                        born_inside_parent *= (Joo[i,j]<=Too_cont_max)
                        born_inside_parent *= (Ntracer_Ac[i]>=Tnc_parent_factor*Ntracer_Bc[j])
                        if born_inside_parent:
                            # Prolly physically interesting new fragmentation
                            nodelabels_Bin[j] = NodeType.IN_BORN_INPARENT
                            edgelabels_Bin[i,j] = EdgeType.IN_PARENT_BIRTH
                        else:
                            nodelabels_Bin[j] = NodeType.IN_RESEGMENTATION
                            edgelabels_Bin[i,j] = EdgeType.IN_RESEGMENTATION
                    # XXX WHAT ABOUT ACCRETION OF SMALL PREDECESSORS?
                else: # (non-tiny flow but no accepted predecessor)
                    nodelabels_Bin[j] = NodeType.IN_BORN_CREATED

        if debug>0: print(f'B_{j}: {NodeType(nodelabels_Bin[j]).name}')

    assert np.all(nodelabels_Bin!=NodeType.UNASSIGNED) # Guard against gaps in the decision tree

    return nodelabels_Bin, edgelabels_Bin


#===============================================================================
# ==== FOPWARD OBJECT RELATION GRAPH ===+=======================================
#===============================================================================

def RelationGraphF(fracmatrices, trcounts, objmasks_R, debug=True):
    ''' 
    Build a relation graph: Sort edges connecting any dendrogram node A_i
    of the previous step to each dendrogram node B_i in the current step. 
    '''
    # ------------------------------------------------------
    # Unpack tracer overlap matrices
    # ------------------------------------------------------
    # Explanation:
    # * c and o subscripts correspond to contained and owned tracers.
    #   A tracer is owned by a dendrogram node if it is directly contained,
    #   i.e. there is no contained subnode that in turn contains the tracer.
    # * Ntracer are the tracer counts for each Ai or Bj, respectively.
    # * F,R and C are forward, reverse and jaccard fractions, i.e.
    #   which fraction of Ai is found in Bj (Forward),
    #   which fraction of Bj is found in Ai (Reverse),
    #   and to which fraction do  Ai and Bj agree (cut/union) (Jaccard).
    # * N are the absolute (non-fractional) tracer counts for the corresponding (Ai,Bj) tracer set unions.
    Ntracer_Ac, Ntracer_Bc, Ntracer_Ao, Ntracer_Bo = trcounts
    Fcc, Rcc, Jcc, Fco, Rco, Jco, Foc, Roc, Joc, Foo, Roo, Joo = fracmatrices
    nmatrices = OverlapCountMatrices(fracmatrices, trcounts)
    Ncc, Nco, Noc, Noo = nmatrices
    nodedepth_R = cztrown.GetNodeDepth(objmasks_R)

    # ------------------------------------------------------
    # --- Prepare result fields
    # ------------------------------------------------------
    # Edge label matrix expressing type of each possible
    # edge from Ai to Bj in form of an integer enumerating the edge type.
    # We will initialize with 0 (not assigned).
    dshape = Fcc.shape
    nA, nB = dshape
    edgelabels_Aout = np.full(dshape, EdgeType.UNASSIGNED, dtype=np.int32)
    nodelabels_Aout = np.full((nA,), NodeType.UNASSIGNED, dtype=np.int32)

    # --- Check that input data contains enough nodes to work on.
    #     The following code is hopeless and will go wrong if not...
    try:
        assert nA>2
        assert nB>2
    except AssertionError:
        print(f'WARNING: RelationGraph called for {nA}x{nB} (<3) nodes.')
        return nodelabels_Aout, edgelabels_Aout

    # ------------------------------------------------------
    # --- Find relevant connections
    # ------------------------------------------------------
    #     Build sparse edge set using shared-tracer minimum and fraction thresholds.
    # Identity edge candidates: Ai and Bj might be the same node
    id_candidates = (Noo>Noo_min)*((Joo>TJoo_id)|(Foo>TFoo_id)|(Roo>TRoo_id))
    # Material flow edge candidate
    flow_candidates = (Ncc>Ncc_min)*((Jcc>TJcc_flow)|(Fcc>TFcc_flow)|(Rcc>TRcc_flow))
    
    # --- Primary sucessor / predecessor:
    #     Find mutual-best owned-owned matches as primary continuations
    # Primary identity successors/predecessors
    s_id_A = np.argmax(Joo, axis=1) # Which Bj is most probable identity sucessor of Ai? (Index in B)
    p_id_B = np.argmax(Joo, axis=0) # Which Ai is most probable identity predecessor of Bj? (Index in A)
    # Mutual-best pair matches:
    # Where does s_id_A match up with p_id_B, i.e. s_id_A[i]==j and p_id_B[j]==i ?
    i_,_j = np.meshgrid(np.arange(nA),np.arange(nB),indexing='ij')
    id_pairs = (s_id_A[:,None]==_j)*(p_id_B[None,:]==i_) # XXX THIS MIGHT BE WRONG XXX
    id_best = id_candidates * id_pairs

    # --- Id flow diagnostics
    # This will later be used to verify 1-to-1 object identities.
    # Note that we normalize under the identity mask to disqualify multiple strong Fcc or Rcc matches.
    # Normalize outgoing flow weights Fcc
    S_out = np.sum(Fcc*id_candidates, axis=1)+1e-10
    Ps_out = Fcc*id_candidates / S_out[:,None]
    # Normalize incoming flow weights Rcc
    S_in = np.sum(Rcc*id_candidates, axis=0)+1e-10
    Pp_in = Rcc*id_candidates / S_in[None,:]

    # ------------------------------------------------------
    # --- Decision tree (Nodes Ai, outgoing)
    # ------------------------------------------------------
    for i in range(nA):
        # * A. GET RAW SUCCESSOR CANDIDATES
        M_flow_successors_raw = flow_candidates[i,:]
        J_flow_successors_raw = np.where(M_flow_successors_raw)[0]
        degree_out_raw = np.sum(M_flow_successors_raw)
        # * B. DO HIERARCHY PRUNING OF FLOW SUCCESSOR CANDIDATES
        if degree_out_raw > 0:
            lineage_groups = GroupByLinage(J_flow_successors_raw, objmasks_R)
            J_successors = [ChooseBestSuccessorLexi(i, Jg, fracmatrices, nmatrices, nodedepth_R) for Jg in lineage_groups]
            assert len(J_successors) == len(set(J_successors)) # Sanity check: No double j from algo
        else:
            J_successors = list()
        M_successors = np.zeros(nB, dtype=bool)
        M_successors[J_successors] = True
        degree_out = len(J_successors)
        # Sanity checks
        assert len(J_successors) == np.sum(M_successors)
        assert np.all(M_successors*flow_candidates[i,:] == M_successors)
        if degree_out_raw > 0: assert degree_out > 0
        # * C. CALCULATE DIAGNOSTICS/WEIGHTS AFTER PRUNING
        # Successor weights 
        succ_weights_out = NormalizeWeights(Fcc[i,:], M_successors)
        succ1_out, succ2_out = np.sort(succ_weights_out)[[-1,-2]]
        if not succ2_out < .5:
            print(f'WARNING: succ1_out={succ1_out}, succ2_out={succ2_out}>=0.5')
        # Index of best successor (will be random if there is none)
        jw1_out = np.argmax(succ_weights_out)
        # Outgoing normalized entropy (Note that this will sum to 0 for degree==0):
        Pf_out = succ_weights_out[M_successors]
        assert np.all(Pf_out>0.)
        H_out = -np.sum(Pf_out*np.log(Pf_out)) / np.log(max(2,degree_out))

        # ------------------------------------------------------    
        # * Path 1: Ai has a strong mutual-best identity successor
        # ------------------------------------------------------    
        if np.any(id_best[i,:]):
            assert np.sum(id_best[i,:]) == 1
            j = np.argmax(id_best[i,:])
            # * Step 2: Is the material flow (i.e. contained material) also mostly identical?
            one_to_one_match  = (Fcc[i,j]>TFcc_1to1)
            one_to_one_match *= (Rcc[i,j]>TRcc_1to1)
            one_to_one_match *= (Ps_out[i,j]>T_dom_out) # XXX Really necessary?
            one_to_one_match *= (Pp_in[i,j]>T_dom_in) # XXX Really necessary?
            if one_to_one_match:
                wj = np.where((Foo)[i,:]>T_minor_drain)[0]
                wjj = wj[wj!=j] # keep all but j itself
                extra_mass_drain = len(wjj)>1 # == 1 would cover the match itself
                if extra_mass_drain:
                    nodelabels_Aout[i] = NodeType.OUT_CONTINUED_DRAIN
                    edgelabels_Aout[i,wjj] = EdgeType.OUT_CONT_DRAIN
                    edgelabels_Aout[i,j] = EdgeType.OUT_CONTINUED_MAIN 
                else:
                    # TODO: Optional distinction: continue+grow, continue+shrink, continue+restructure
                    nodelabels_Aout[i] = NodeType.OUT_CONTINUED
                    edgelabels_Aout[i,j] = EdgeType.OUT_CONTINUED_MAIN
            else:
                shedding = (degree_out>=2) * (succ1_out>T_dom_split) * (succ2_out>T_2_split)
                split    = (degree_out>=2) * (succ1_out<=T_dom_split) * (succ2_out>T_2_split)
                if shedding:
                    # One persistent main continuation plus side branches
                    nodelabels_Aout[i] = NodeType.OUT_CONTINUED_SHEDDING
                    # ORDER MATTERS: PRIORITY OVERWRITE: OUTFLOW < SHEDDING < MAIN ROUTE
                    wj = np.where(succ_weights_out>T_minor_drain)[0]
                    edgelabels_Aout[i,wj] = EdgeType.OUT_CONT_DRAIN
                    wj = np.where(succ_weights_out>T_2_split)[0]
                    edgelabels_Aout[i,wj] = EdgeType.OUT_CONT_SHEDDING
                    edgelabels_Aout[i,j] = EdgeType.OUT_CONTINUED_MAIN                
                elif split:
                    # Split without dominant flow but with several weak sucessors
                    nodelabels_Aout[i] = NodeType.OUT_CONTINUATION_SPLIT
                    wj = np.where(succ_weights_out>T_2_split)[0]
                    edgelabels_Aout[i,wj] = EdgeType.OUT_CONT_SPLIT2
                    edgelabels_Aout[i,j] = EdgeType.OUT_CONT_SPLIT1
                else:
                    wj = np.where(id_candidates[i,:])[0]
                    wjj = wj[wj!=j]
                    if debug > 0:
                        print('IN_CONTINUATION_AMBIGUOUS:')
                        print(f'(Fcc[{i},{j}]>TFcc_1to1): {(Fcc[i,j]>TFcc_1to1)} # {Fcc[i,j]} > {TFcc_1to1} ?')
                        print(f'(Rcc[{i},{j}]>TRcc_1to1): {(Rcc[i,j]>TRcc_1to1)} # {Rcc[i,j]} > {TRcc_1to1} ?')
                        print(f'(Ps_out[{i},{j}]>T_dom_out): {(Ps_out[i,j]>T_dom_out)} # {Ps_out[i,j]} > {T_dom_out} ?')
                        print(f'(Pp_in[{i},{j}]>T_dom_in): {(Pp_in[i,j]>T_dom_in)} # {Pp_in[i,j]} > {T_dom_in} ?')
                        print(f'Best candidate: B{j}')
                        print(f'Alternative candidates: B{wjj}')
                    if len(wjj) > 0:
                        nodelabels_Aout[i] = NodeType.OUT_CONTINUATION_AMBIGUOUS_MULTI
                    else:
                        nodelabels_Aout[i] = NodeType.OUT_CONTINUATION_AMBIGUOUS_WEAK
                    edgelabels_Aout[i,wjj] = EdgeType.OUT_CONT_AMBIGUOUS_ALT
                    edgelabels_Aout[i,j] = EdgeType.OUT_CONT_AMBIGUOUS_BEST
                    
        # ------------------------------------------------------    
        # * Path 2: Ai has no mutual best successor
        # ------------------------------------------------------    
        else:
            # XXX MAYBE NEED TO CHANGE LOGIC:
            # * NEED BEST FLOW EQUIV. (CC-SUCCESSOR) FOR MERGERS
            # * NEED BEST CONTAINER (MIXED-SUCCESSOR (WHICH ONE? OC?) IN_PARENT STUFF
            forward_flow_tiny = (np.sum(Fcc[i,:])<=T_vanish_max)
            if forward_flow_tiny:
                # No strong continuation and no significant flow successor
                nodelabels_Aout[i] = NodeType.OUT_LOST_VANISHED
            else:
                many_weak_successors = (degree_out>2)*(H_out>H_high)
                has_any_successor = (degree_out>0)
                if many_weak_successors:
                    nodelabels_Aout[i] = NodeType.OUT_LOST_DISPERSED
                    edgelabels_Aout[i,J_successors] = EdgeType.OUT_WEAK_SUCCESSORS
                elif has_any_successor:
                    # Strong material flow into one node j?
                    j = jw1_out
                    # * Step 6: True split (i.e. several non-dominant but substantial successors)?
                    ### Note: The (succ1_out<=T_dom_split) condition distinguishes true splitting from
                    #         mass transfer by abrasion (resegmentation)
                    true_split = (degree_out>=2) * (succ1_out<=T_dom_split) * (succ2_out>T_2_split)
                    if true_split:
                        wj = np.where(succ_weights_out>T_2_split)[0]
                        edgelabels_Aout[i,wj] = EdgeType.OUT_SPLIT2
                        edgelabels_Aout[i,j] = EdgeType.OUT_SPLIT1
                        if (degree_out==2):
                            nodelabels_Aout[i] = NodeType.OUT_SPLIT_BINARY
                        elif (degree_out>=3):
                            nodelabels_Aout[i] = NodeType.OUT_SPLIT_MULTI
                    else: # (not a true split)
                        # * Step 7: Absorbed into parent / junior partner in merger # XXX WHAT IS IT THEH, WTF??? XXX
                        # XXX BUG ? WHY IS THIS IN THE has_any_successor SECTION? THIS IS PROBABLY WRONG!
                        absorbed_into_parent  = (Rco[i,j]>Tco_absorp_parent) | (Roc[i,j]>Toc_absorp_parent)
                        absorbed_into_parent *= (Joo[i,j]<=Too_cont_max)
                        absorbed_into_parent *= (Ntracer_Bc[j]>=Tnc_parent_factor*Ntracer_Ac[i])
                        if absorbed_into_parent:
                            nodelabels_Aout[i] = NodeType.OUT_ABSORP_INPARENT
                            edgelabels_Aout[i,j] = EdgeType.OUT_CHILD_DEATH
                        else:
                            nodelabels_Aout[i] = NodeType.OUT_RESEGMENTATION
                            edgelabels_Aout[i,j] = EdgeType.OUT_RESEGMENTATION
                else: # (non-tiny flow but no accepted successor)
                    nodelabels_Aout[i] = NodeType.OUT_LOST_DESTROYED

        if debug>0: print(f'A_{i}: {NodeType(nodelabels_Aout[i]).name}')

    assert np.all(nodelabels_Aout!=NodeType.UNASSIGNED) # Guard against gaps in the decision tree

    return nodelabels_Aout, edgelabels_Aout


#===============================================================================
# ==== DATA HELPER =============================================================
#===============================================================================

def OverlapCountMatrices(fracmatrices, trcounts):
    # Unpack data
    Fcc, Rcc, Jcc, Fco, Rco, Jco, Foc, Roc, Joc, Foo, Roo, Joo = fracmatrices
    N1c, N2c, N1o, N2o = trcounts
    # Calculate back from fractional overlap to overlap tracer count (forward)
    Ncc_F = np.astype(Fcc*N1c[:,None]+.5, int)
    Nco_F = np.astype(Fco*N1c[:,None]+.5, int)
    Noc_F = np.astype(Foc*N1o[:,None]+.5, int)
    Noo_F = np.astype(Foo*N1o[:,None]+.5, int)
    # Calculate back from fractional overlap to overlap tracer count (reverse)
    Ncc_R = np.astype(Rcc*N2c[None,:]+.5, int)
    Nco_R = np.astype(Rco*N2o[None,:]+.5, int)
    Noc_R = np.astype(Roc*N2c[None,:]+.5, int)
    Noo_R = np.astype(Roo*N2o[None,:]+.5, int)
    # Check the result for validity
    assert np.all(Ncc_F==Ncc_R)
    assert np.all(Nco_F==Nco_R)
    assert np.all(Noc_F==Noc_R)
    assert np.all(Noo_F==Noo_R)
    # Return forward version
    return Ncc_F, Nco_F, Noc_F, Noo_F


#===============================================================================
# ==== TEST EDGES ==============================================================
#===============================================================================

matrix_names = ['Mcc_f', 'Mcc_r', 'Mcc_j', 'Mco_f', 'Mco_r', 'Mco_j', 'Moc_f', 'Moc_r', 'Moc_j', 'Moo_f', 'Moo_r', 'Moo_j']
matrix_order = [0,3,6,9, 1,4,7,10, 2,5,8,11,]

Values = lambda a: [elem.item() for elem in a]
fValues = lambda a: [f'{elem.item():.4f}' for elem in a]

def ShowObjMatricesF(matrices, iA):
    M_iA = np.array(matrices)[:,iA,:]
    jmask = np.max(M_iA, axis=0) > 1e-5
    J = np.arange(M_iA.shape[-1])
    J_ = Values(J[jmask])
    print(f'J-indicies: {J_}')
    for k in matrix_order:
        matline = M_iA[k,jmask]
        matname = matrix_names[k]
        print(f'{matname}[{iA},J] = {fValues(matline)}')

def ShowObjMatricesR(matrices, jB):
    M_jB = np.array(matrices)[:,:,jB]
    imask = np.max(M_jB, axis=0) > 1e-5
    I = np.arange(M_jB.shape[-1])
    I_ = Values(I[imask])
    print(f'I-indicies: {I_}')
    for k in matrix_order:
        matcol = M_jB[k,imask]
        matname = matrix_names[k]
        print(f'{matname}[I,{jB}] = {fValues(matcol)}')

def ShowNodeLabels(nodelabels):
    for inode, enlabel in enumerate(nodelabels):
        print(f'Node {inode}: {NodeType(enlabel).name}')

def CheckContinuationSymmetryAB(nodelabels_Aout, edgelabels_Aout, nodelabels_Bin, edgelabels_Bin):
    condition = [nt in (NodeType.OUT_CONTINUED, NodeType.OUT_CONTINUED_DRAIN, NodeType.OUT_CONTINUED_SHEDDING) for nt in nodelabels_Aout]
    print(f'--- CONTINUATION OUT --- ({np.sum(condition)})')
    for iA, Alabel in enumerate(nodelabels_Aout):
        if condition[iA]:
            print(f'A_{iA}:{NodeType(Alabel).name} -> ', end='')
            M = (edgelabels_Aout[iA,:]==EdgeType.OUT_CONTINUED_MAIN)
            assert np.sum(M) == 1
            j_cont = np.where(M)[0].item()
            print(f'B_{j_cont}:{NodeType(nodelabels_Bin[j_cont]).name} -< ', end='')
            M = (edgelabels_Bin[:,j_cont]==EdgeType.IN_CONTINUED_MAIN)
            assert np.sum(M) <= 1
            if np.sum(M):
                i_cont = np.where(M)[0].item()
                print(f'A_{i_cont}')
                assert(iA==i_cont)
            else:
                print('X')

def CheckContinuationSymmetryBA(nodelabels_Aout, edgelabels_Aout, nodelabels_Bin, edgelabels_Bin):
    condition = [nt in (NodeType.IN_CONTINUED, NodeType.IN_CONTINUED_ACCRETION, NodeType.IN_CONTINUED_CAPTURE) for nt in nodelabels_Bin]
    print(f'--- CONTINUATION IN --- ({np.sum(condition)})')
    for jB, Blabel in enumerate(nodelabels_Bin):
        if condition[jB]:
            print(f'B_{jB}:{NodeType(Blabel).name} -< ', end='')
            M = (edgelabels_Bin[:,jB]==EdgeType.IN_CONTINUED_MAIN)
            assert np.sum(M) == 1
            i_cont = np.where(M)[0].item()
            print(f'A_{i_cont}:{NodeType(nodelabels_Aout[i_cont]).name} -> ', end='')
            M = (edgelabels_Aout[i_cont,:]==EdgeType.OUT_CONTINUED_MAIN)
            assert np.sum(M) <= 1
            if np.sum(M):
                j_cont = np.where(M)[0].item()
                print(f'B_{j_cont}')
                assert(jB==j_cont)
            else:
                print('X')

def CheckSplitAB(nodelabels_Aout, edgelabels_Aout, nodelabels_Bin, edgelabels_Bin):
    condition = [nt in (NodeType.OUT_CONTINUATION_SPLIT, NodeType.OUT_SPLIT_BINARY, NodeType.OUT_SPLIT_MULTI) for nt in nodelabels_Aout]
    print(f'--- SPLITS --- ({np.sum(condition)})')
    for iA, Alabel in enumerate(nodelabels_Aout):
        if condition[iA]:
            print(f'A_{iA}:{NodeType(Alabel).name} -> ', end='')
            M1 = (edgelabels_Aout[iA,:]==EdgeType.OUT_CONT_SPLIT1) | (edgelabels_Aout[iA,:]==EdgeType.OUT_SPLIT1)
            M2 = (edgelabels_Aout[iA,:]==EdgeType.OUT_CONT_SPLIT2) | (edgelabels_Aout[iA,:]==EdgeType.OUT_SPLIT2)
            j1 = np.where(M1)[0].item()
            J2 = np.where(M2)[0]
            print(f'B_({j1}|{J2})...')
            print(f'* B_{j1}:{NodeType(nodelabels_Bin[j1]).name} -- E:{EdgeType(edgelabels_Bin[iA,j1]).name}')
            for j2 in J2:
                print(f'+ B_{j2}:{NodeType(nodelabels_Bin[j2]).name} -- E:{EdgeType(edgelabels_Bin[iA,j2]).name}')
            print()

def CheckMergerBA(nodelabels_Aout, edgelabels_Aout, nodelabels_Bin, edgelabels_Bin):
    condition = [nt in (NodeType.IN_CONTINUATION_MERGER, NodeType.IN_MERGE_BINARY, NodeType.IN_MERGE_MULTI) for nt in nodelabels_Bin]
    print(f'--- MERGER --- ({np.sum(condition)})')
    for jB, Blabel in enumerate(nodelabels_Bin):
        if condition[jB]:
            print(f'B_{jB}:{NodeType(Blabel).name} -< ', end='')
            M1 = (edgelabels_Bin[:,jB]==EdgeType.IN_CONT_MERGE1) | (edgelabels_Bin[:,jB]==EdgeType.IN_MERGE1)
            M2 = (edgelabels_Bin[:,jB]==EdgeType.IN_CONT_MERGE2) | (edgelabels_Bin[:,jB]==EdgeType.IN_MERGE2)
            i1 = np.where(M1)[0].item()
            I2 = np.where(M2)[0]
            print(f'A_({i1}|{I2})...')
            print(f'* A_{i1}:{NodeType(nodelabels_Aout[i1]).name} -- E:{EdgeType(edgelabels_Aout[i1,jB]).name}')
            for i2 in I2:
                print(f'+ A_{i2}:{NodeType(nodelabels_Aout[i2]).name} -- E:{EdgeType(edgelabels_Aout[i2,jB]).name}')
            print()

def CheckSheddingAB(nodelabels_Aout, edgelabels_Aout, nodelabels_Bin, edgelabels_Bin):
    condition = [nt == NodeType.OUT_CONTINUED_SHEDDING for nt in nodelabels_Aout]
    print(f'--- SHEDDING --- ({np.sum(condition)})')
    for iA, Alabel in enumerate(nodelabels_Aout):
        if condition[iA]:
            print(f'A_{iA}:{NodeType(Alabel).name} -> ', end='')
            M1 = (edgelabels_Aout[iA,:]==EdgeType.OUT_CONTINUED_MAIN)
            M2 = (edgelabels_Aout[iA,:]==EdgeType.OUT_CONT_SHEDDING)
            j1 = np.where(M1)[0].item()
            J2 = np.where(M2)[0]
            print(f'B_({j1}|{J2})...')
            print(f'* B_{j1}:{NodeType(nodelabels_Bin[j1]).name} -- E:{EdgeType(edgelabels_Bin[iA,j1]).name}')
            for j2 in J2:
                print(f'+ B_{j2}:{NodeType(nodelabels_Bin[j2]).name} -- E:{EdgeType(edgelabels_Bin[iA,j2]).name}')
            print()

def CheckCaptureBA(nodelabels_Aout, edgelabels_Aout, nodelabels_Bin, edgelabels_Bin):
    condition = [nt == NodeType.IN_CONTINUED_CAPTURE for nt in nodelabels_Bin]
    print(f'--- CAPTURE --- ({np.sum(condition)})')
    for jB, Blabel in enumerate(nodelabels_Bin):
        if condition[jB]:
            print(f'B_{jB}:{NodeType(Blabel).name} -< ', end='')
            M1 = (edgelabels_Bin[:,jB]==EdgeType.IN_CONTINUED_MAIN)
            M2 = (edgelabels_Bin[:,jB]==EdgeType.IN_CONT_CAPTURE)
            i1 = np.where(M1)[0].item()
            I2 = np.where(M2)[0]
            print(f'A_({i1}|{I2})...')
            print(f'* A_{i1}:{NodeType(nodelabels_Aout[i1]).name} -- E:{EdgeType(edgelabels_Aout[i1,jB]).name}')
            for i2 in I2:
                print(f'+ A_{i2}:{NodeType(nodelabels_Aout[i2]).name} -- E:{EdgeType(edgelabels_Aout[i2,jB]).name}')
            print()

def CheckLostAB(nodelabels_Aout, edgelabels_Aout, nodelabels_Bin, edgelabels_Bin):
    condition = [nt in (NodeType.OUT_LOST_VANISHED, NodeType.OUT_LOST_DISPERSED, NodeType.OUT_ABSORP_INPARENT,
                        NodeType.OUT_RESEGMENTATION, NodeType.OUT_LOST_DESTROYED) for nt in nodelabels_Aout]
    print(f'--- LOST --- ({np.sum(condition)})')
    for iA, Alabel in enumerate(nodelabels_Aout):
        if condition[iA]:
            print(f'A_{iA}:{NodeType(Alabel).name}')
            for jB, label_Bin in enumerate(edgelabels_Bin[iA,:]):
                label_Aout = edgelabels_Aout[iA,jB]
                if (EdgeType(label_Bin) != EdgeType.UNASSIGNED) or (EdgeType(label_Aout) != EdgeType.UNASSIGNED):
                    print(f'-EF_[{iA},{jB}]:{EdgeType(label_Aout).name}')
                    print(f'-ER_[{iA},{jB}]:{EdgeType(label_Bin).name}')
                    print(f'>B_{jB}:{NodeType(nodelabels_Bin[jB]).name}')
                    print('-')
            print()
    

def CheckCreatedBA(nodelabels_Aout, edgelabels_Aout, nodelabels_Bin, edgelabels_Bin):
    condition = [nt in (NodeType.IN_BORN_NOISY, NodeType.IN_BORN_DIFFUSE, NodeType.IN_BORN_INPARENT,
                        NodeType.IN_RESEGMENTATION, NodeType.IN_BORN_CREATED) for nt in nodelabels_Bin]
    print(f'--- BORN --- ({np.sum(condition)})')
    for jB, Blabel in enumerate(nodelabels_Bin):
        if condition[jB]:
            print(f'B_{jB}:{NodeType(Blabel).name}')
            for iA, label_Aout in enumerate(edgelabels_Aout[:,jB]):
                label_Bin = edgelabels_Bin[iA,jB]
                if (EdgeType(label_Bin) != EdgeType.UNASSIGNED) or (EdgeType(label_Aout) != EdgeType.UNASSIGNED):
                    print(f'-ER_[{iA},{jB}]:{EdgeType(label_Bin).name}')
                    print(f'-EF_[{iA},{jB}]:{EdgeType(label_Aout).name}')
                    print(f'<A_{iA}:{NodeType(nodelabels_Aout[iA]).name}')
                    print('-')
            print()


#===============================================================================
# ==== TEST MAIN ==============================================================
#===============================================================================

ppath = '/home/mweis/scratch/thesisdata/corezoom-work-wbox/b.Aa1-Aa4-2zone/out'
pf1 = os.path.join(ppath, 'CZ-b_hdf5_plt_cnt_0305')
pf2 = os.path.join(ppath, 'CZ-b_hdf5_plt_cnt_0306')
#pf3 = os.path.join(ppath, 'CZ-b_hdf5_plt_cnt_0203')
pd1 = cz.PartfileFromPlotfile(pf1)
pd2 = cz.PartfileFromPlotfile(pf2)
#pd3 = cz.PartfileFromPlotfile(pf3)
criterion = 'COCxp30:adrnode:czfrag'

def Test():
    leafs1, oct1 = cz.LoadPlotfile(pf1)
    objm1 = clfind.get_cmasks(leafs1, oct1, criterion)
    leafs2, oct2 = cz.LoadPlotfile(pf2)
    objm2 = clfind.get_cmasks(leafs2, oct2, criterion)
    trmatch12 = cztrmatch.get_matchmatrix(leafs1,oct1, leafs2,oct2, criterion)
    matrices12, trcounts12 = trmatch12
    #
    print()
    print('-> Computing RelationGraphR...')
    sys.stdout.flush()
    nodelabels_Bin, edgelabels_Bin = RelationGraphR(matrices12, trcounts12, objm1)
    #
    print()
    print('-> Computing RelationGraphF...')
    sys.stdout.flush()
    nodelabels_Aout, edgelabels_Aout = RelationGraphF(matrices12, trcounts12, objm2)
    #print(f'--- NODE LABELS B:IN ---')
    #ShowNodeLabels(nodelabels_Bin)
    #print()
    #print(f'--- NODE LABELS A:OUT ---')
    #ShowNodeLabels(nodelabels_Aout)
    #print()
    #CheckContinuationSymmetryAB(nodelabels_Aout, edgelabels_Aout, nodelabels_Bin, edgelabels_Bin)
    #print()
    #CheckContinuationSymmetryBA(nodelabels_Aout, edgelabels_Aout, nodelabels_Bin, edgelabels_Bin)
    #print()
    #CheckSplitAB(nodelabels_Aout, edgelabels_Aout, nodelabels_Bin, edgelabels_Bin)
    #print()
    #CheckMergerBA(nodelabels_Aout, edgelabels_Aout, nodelabels_Bin, edgelabels_Bin)
    #print()
    #CheckSheddingAB(nodelabels_Aout, edgelabels_Aout, nodelabels_Bin, edgelabels_Bin)
    #print()
    #CheckCaptureBA(nodelabels_Aout, edgelabels_Aout, nodelabels_Bin, edgelabels_Bin)
    #print()
    #CheckLostAB(nodelabels_Aout, edgelabels_Aout, nodelabels_Bin, edgelabels_Bin)
    #print()
    #CheckCreatedBA(nodelabels_Aout, edgelabels_Aout, nodelabels_Bin, edgelabels_Bin)
    #edgelabels = InterpretEdges(edgelabels_Aout, edgelabels_Bin)
    #CheckEdgeFunnies(nodelabels_Aout, nodelabels_Bin, edgelabels, edgelabels_Aout, edgelabels_Bin)
    ##
    ##leafs3, oct3 = cz.LoadPlotfile(pf3)
    ##objm3 = clfind.get_cmasks(leafs2, oct2, criterion)
    ##trmatch23 = cztrmatch.get_matchmatrix(leafs1,oct1, leafs2,oct2, criterion)
    ##matrices23, (n_tracer2b,n_tracer3) = trmatch23
    ##
    #print()
    #Fcc, Rcc, Jcc, Fco, Rco, Jco, Foc, Roc, Joc, Foo, Roo, Joo = matrices12
    #print(f'--- LOST & FOUND ---')
    #for i in (16,31):
    #    print('--')
    #    for j in (3,31,37,39):
    #        print('-')
    #        print(f'Fcc[{i},{j}] = {Fcc[i,j]}')
    #        print(f'Rcc[{i},{j}] = {Rcc[i,j]}')
    #        print(f'Foo[{i},{j}] = {Foo[i,j]}')
    #        print(f'Roo[{i},{j}] = {Roo[i,j]}')
    #
    #pass

#===============================================================================
# ==== LOBBY ===================================================================
#===============================================================================

if __name__ == '__main__':
    Test()
