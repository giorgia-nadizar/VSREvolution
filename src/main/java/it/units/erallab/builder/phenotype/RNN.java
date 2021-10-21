package it.units.erallab.builder.phenotype;

import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.hmsrobots.core.controllers.RecurrentNeuralNetwork;
import it.units.erallab.hmsrobots.core.controllers.TimedRealFunction;

import java.util.Collections;
import java.util.List;
import java.util.function.Function;

/**
 * @author eric
 */
public class RNN implements PrototypedFunctionBuilder<List<Double>, TimedRealFunction> {

  private final int nOfRecurrentNeurons;

  public RNN(int nOfRecurrentNeurons) {
    this.nOfRecurrentNeurons = nOfRecurrentNeurons;
  }

  @Override
  public Function<List<Double>, TimedRealFunction> buildFor(TimedRealFunction function) {
    return values -> {
      int nOfInputs = function.getInputDimension();
      int nOfOutputs = function.getOutputDimension();
      int nOfWeights = RecurrentNeuralNetwork.countWeights(nOfInputs, nOfRecurrentNeurons, nOfOutputs);
      if (nOfWeights != values.size()) {
        throw new IllegalArgumentException(String.format(
            "Wrong number of values for weights: %d expected, %d found",
            nOfWeights,
            values.size()
        ));
      }
      return new RecurrentNeuralNetwork(
          nOfInputs,
          nOfRecurrentNeurons,
          nOfOutputs,
          values.stream().mapToDouble(d -> d).toArray()
      );
    };
  }

  @Override
  public List<Double> exampleFor(TimedRealFunction function) {
    return Collections.nCopies(
        RecurrentNeuralNetwork.countWeights(
            function.getInputDimension(),
            nOfRecurrentNeurons,
            function.getOutputDimension()),
        0d
    );
  }

}
