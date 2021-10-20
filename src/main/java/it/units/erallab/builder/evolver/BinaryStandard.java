package it.units.erallab.builder.evolver;

import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.malelab.jgea.core.Individual;
import it.units.malelab.jgea.core.evolver.Evolver;
import it.units.malelab.jgea.core.evolver.StandardEvolver;
import it.units.malelab.jgea.core.evolver.StandardWithEnforcedDiversityEvolver;
import it.units.malelab.jgea.core.order.PartialComparator;
import it.units.malelab.jgea.core.selector.Last;
import it.units.malelab.jgea.core.selector.Tournament;
import it.units.malelab.jgea.representation.sequence.SameTwoPointsCrossover;
import it.units.malelab.jgea.representation.sequence.bit.BitFlipMutation;
import it.units.malelab.jgea.representation.sequence.bit.BitString;
import it.units.malelab.jgea.representation.sequence.bit.BitStringFactory;

import java.util.Map;

/**
 * @author eric
 */
public class BinaryStandard implements EvolverBuilder<BitString> {

  private final int nPop;
  private final int nTournament;
  private final double xOverProb;
  protected final boolean diversityEnforcement;

  public BinaryStandard(int nPop, int nTournament, double xOverProb, boolean diversityEnforcement) {
    this.nPop = nPop;
    this.nTournament = nTournament;
    this.xOverProb = xOverProb;
    this.diversityEnforcement = diversityEnforcement;
  }

  @Override
  public <T, F> Evolver<BitString, T, F> build(PrototypedFunctionBuilder<BitString, T> builder, T target, PartialComparator<F> comparator) {
    int length = builder.exampleFor(target).size();
    if (!diversityEnforcement) {
      return new StandardEvolver<>(
          builder.buildFor(target),
          new BitStringFactory(length),
          comparator.comparing(Individual::getFitness),
          nPop,
          Map.of(
              new BitFlipMutation(.01d), 1d - xOverProb,
              new SameTwoPointsCrossover<>(new BitStringFactory(length))
                  .andThen(new BitFlipMutation(.01d)), xOverProb
          ),
          new Tournament(nTournament),
          new Last(),
          nPop,
          true,
          true
      );
    }
    return new StandardWithEnforcedDiversityEvolver<>(
        builder.buildFor(target),
        new BitStringFactory(length),
        comparator.comparing(Individual::getFitness),
        nPop,
        Map.of(
            new BitFlipMutation(.35d), 1d - xOverProb,
            new SameTwoPointsCrossover<>(new BitStringFactory(length))
                .andThen(new BitFlipMutation(.1d)), xOverProb
        ),
        new Tournament(nTournament),
        new Last(),
        nPop,
        true,
        true,
        100
    );
  }

}
