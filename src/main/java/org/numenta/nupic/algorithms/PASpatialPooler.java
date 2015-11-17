/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2014, Numenta, Inc.  Unless you have an agreement
 * with Numenta, Inc., for a separate license for this software code, the
 * following terms and conditions apply:
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero Public License version 3 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU Affero Public License for more details.
 *
 * You should have received a copy of the GNU Affero Public License
 * along with this program.  If not, see http://www.gnu.org/licenses.
 *
 * http://numenta.org/licenses/
 * ---------------------------------------------------------------------
 */

package org.numenta.nupic.algorithms;

import gnu.trove.list.array.TIntArrayList;
import org.numenta.nupic.Connections;
import org.numenta.nupic.model.Column;
import org.numenta.nupic.util.ArrayUtils;

import java.util.*;

/**
 * Subclasses {@link SpatialPooler} to perform Prediction-Assisted CLA
 *
 * @author David Ray
 * @author Fergal Byrne
 *
 */
public class PASpatialPooler extends SpatialPooler {
    /**
     * Step two of pooler initialization kept separate from initialization
     * of static members so that they may be set at a different point in
     * the initialization (as sometimes needed by tests).
     *
     * This step prepares the proximal dendritic synapse pools with their
     * initial permanence values and connected inputs.
     *
     * @param c		the {@link Connections} memory
     */
    public void connectAndConfigureInputs(Connections c) {
        // Initialize the set of permanence values for each column. Ensure that
        // each column is connected to enough input bits to allow it to be
        // activated.
        int numColumns = c.getNumColumns();
        for(int i = 0;i < numColumns;i++) {
            int[] potential;
            Column column = c.getColumn(i);
            double[] perm;
            if(c.getSPOneToOne()) {
                potential = new int[] {i};
                c.getPotentialPools().set(i, column.createPotentialPool(c, potential));
                perm = new double[c.getNumInputs()];
                perm[i] = c.getSynPermConnected() + c.getRandom().nextDouble() * c.getSynPermActiveInc() / 4.0;
                c.getColumn(i).setProximalPermanences(c, perm);
            } else {
                potential = mapPotential(c, i, true);
                c.getPotentialPools().set(i, column.createPotentialPool(c, potential));
                perm = initPermanence(c, potential, i, c.getInitConnectedPct());
                updatePermanencesForColumn(c, perm, column, potential, true);
            }
        }
        updateInhibitionRadius(c);
    }
    /**
     * This is the primary public method of the SpatialPooler class. This
     * function takes a input vector and outputs the indices of the active columns.
     * If 'learn' is set to True, this method also updates the permanences of the
     * columns.
     * @param inputVector       An array of 0's and 1's that comprises the input to
     *                          the spatial pooler. The array will be treated as a one
     *                          dimensional array, therefore the dimensions of the array
     *                          do not have to match the exact dimensions specified in the
     *                          class constructor. In fact, even a list would suffice.
     *                          The number of input bits in the vector must, however,
     *                          match the number of bits specified by the call to the
     *                          constructor. Therefore there must be a '0' or '1' in the
     *                          array for every input bit.
     * @param activeArray       An array whose size is equal to the number of columns.
     *                          Before the function returns this array will be populated
     *                          with 1's at the indices of the active columns, and 0's
     *                          everywhere else.
     * @param learn             A boolean value indicating whether learning should be
     *                          performed. Learning entails updating the  permanence
     *                          values of the synapses, and hence modifying the 'state'
     *                          of the model. Setting learning to 'off' freezes the SP
     *                          and has many uses. For example, you might want to feed in
     *                          various inputs and examine the resulting SDR's.
     */
    public void compute(Connections c, int[] inputVector, int[] activeArray, boolean learn, boolean stripNeverLearned) {
        if(inputVector.length != c.getNumInputs()) {
            throw new IllegalArgumentException(
                    "Input array must be same size as the defined number of inputs: From Params: " + c.getNumInputs() +
                            ", From Input Vector: " + inputVector.length);
        }
        if(c.getSPOneToOne()) {
            int[] overlaps = calculateOverlap(c, inputVector);
            int nActive = 0;
            int nInputsOn = 0;
            int max = 0;


            for(int i = 0; i < overlaps.length; i++) {
                if(inputVector[i] > 0) {
                    nInputsOn++;
                }
                if(overlaps[i] > 0) {
                    nActive++;
                    if(overlaps[i] > max) {
                        max = overlaps[i];
                    }
                }
            }
            int thinnedSDR = Math.min((int) Math.floor((double) nInputsOn * 0.66),
                    (int)Math.floor((double)inputVector.length * 0.01));

            int[] lengths = new int[max];
            int[] levels = new int[nActive * max];
            for(int i = 0; i < overlaps.length; i++) {
                int v = overlaps[i];
                if(v > 0) {
                    levels[(nActive * (v - 1)) + lengths[(v - 1)]] = i;
                    lengths[(v - 1)]++;
                }
            }
            System.out.println("levels: "+levels.length + " " + ArrayUtils.intArrayToString(levels));

            int[] activeColumns = new int[nActive];

            int j = 0;
            for(int i = max - 1; i >= 0 && j < thinnedSDR; i--) {
                if((j + lengths[i]) < thinnedSDR) {
                    for(int k = 0; k < lengths[i]; k++) {
                        activeColumns[j++] = levels[nActive * i + k];
                    }
                } else {
                    int[] choices = Arrays.copyOfRange(levels, nActive * i, nActive * i + lengths[i]);
                    System.out.println("shuffle: [" + j + "/" + thinnedSDR + "] level " + i + ", " +choices.length + " "
                            + ArrayUtils.intArrayToString(choices));
                    ArrayUtils.shuffle(choices);
                    for (int k = 0; k < lengths[i]; k++) {
                        if (j < thinnedSDR) {
                            activeColumns[j++] = choices[k];
                        }
                    }
                }
            }
            /*
            Map<Integer,Set<Integer>> levels = new HashMap<Integer,Set<Integer>>();
            for(int i = 0; i < overlaps.length; i++) {
                if(overlaps[i] > 0) {
                    activeColumns[j++] = i;
                    Set<Integer> indices;
                    Integer iVal = new Integer(overlaps[i]);
                    if(!levels.containsKey(iVal)) {
                        indices = new LinkedHashSet<Integer>();
                        levels.put(iVal,indices);
                        System.out.println("new level: " + iVal.toString());
                    } else {
                        indices = levels.get(iVal);
                    }
                    indices.add(new Integer(i));
                }
            }
            */
            /*
            for(int i = 0; i < overlaps.length; i++) {
                if(overlaps[i] > 0) {
                    activeColumns[j++] = i;
                }
            }
            */
            System.out.println("compute: "+activeColumns.length + " " + ArrayUtils.intArrayToString(activeColumns));
            Arrays.sort(activeColumns);
            Arrays.fill(activeArray, 0);
            if(activeColumns.length > 0) {
                //System.out.println("compute: "+activeColumns.length + " " + ArrayUtils.intArrayToString(activeColumns));
                ArrayUtils.setIndexesTo(activeArray, activeColumns, 1);
            }
            //System.out.println("compute: "+activeArray.length + " on: " + ArrayUtils.sum(activeArray));
        } else {
            super.compute(c, inputVector, activeArray, learn, stripNeverLearned);
        }

    }

    /**
     * This function determines each column's overlap with the current input
     * vector. The overlap of a column is the number of synapses for that column
     * that are connected (permanence value is greater than '_synPermConnected')
     * to input bits which are turned on. Overlap values that are lower than
     * the 'stimulusThreshold' are ignored. The implementation takes advantage of
     * the SpraseBinaryMatrix class to perform this calculation efficiently.
     *
     * @param c				the {@link Connections} memory encapsulation
     * @param inputVector   an input array of 0's and 1's that comprises the input to
     *                      the spatial pooler.
     * @return
     */
    public int[] calculateOverlap(Connections c, int[] inputVector) {
        int[] overlaps = new int[c.getNumColumns()];
        if(c.getSPOneToOne()) {
            overlaps = ArrayUtils.multiply(inputVector, 5);
            //System.out.println(String.format("overlaps[%d], %d on",overlaps.length,ArrayUtils.sum(overlaps)));
        } else {
            c.getConnectedCounts().rightVecSumAtNZ(inputVector, overlaps);
        }
        int[] paOverlaps = ArrayUtils.toIntArray(c.getPAOverlaps());
        overlaps = ArrayUtils.i_add(paOverlaps, overlaps);
        ArrayUtils.lessThanXThanSetToY(overlaps, (int)c.getStimulusThreshold(), 0);
        //if(c.getSPOneToOne()) { System.out.println(String.format("overlaps[%d], %d on",overlaps.length,ArrayUtils.sum(overlaps))); }
        return overlaps;
    }

}
