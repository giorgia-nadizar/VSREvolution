package it.units.erallab.mapper.robot;

import it.units.erallab.hmsrobots.core.controllers.PhaseSin;
import it.units.erallab.hmsrobots.core.objects.ControllableVoxel;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.erallab.mapper.PrototypedFunctionBuilder;

import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;

/**
 * @author eric
 * @created 2020/12/07
 * @project VSREvolution
 */
public class FixedPhaseValues implements PrototypedFunctionBuilder<List<Double>, Robot<?>> {

  private final double frequency;
  private final double amplitude;

  public FixedPhaseValues(double frequency, double amplitude) {
    this.frequency = frequency;
    this.amplitude = amplitude;
  }

  @Override
  public Function<List<Double>, Robot<?>> buildFor(Robot<?> robot) {
    Grid<?> body = robot.getVoxels();
    long nOfVoxel = body.values().stream().filter(Objects::nonNull).count();
    return values -> {
      if (nOfVoxel != values.size()) {
        throw new IllegalArgumentException(String.format(
            "Wrong number of values: %d expected, %d found",
            nOfVoxel,
            values.size()
        ));
      }
      int c = 0;
      Grid<Double> phases = Grid.create(body);
      for (Grid.Entry<?> entry : body) {
        if (entry.getValue() != null) {
          phases.set(entry.getX(), entry.getY(), values.get(c));
          c = c + 1;
        }
      }
      return new Robot<>(
          new PhaseSin(frequency, amplitude, phases),
          (Grid<ControllableVoxel>) SerializationUtils.clone(body)
      );
    };
  }

  @Override
  public List<Double> exampleFor(Robot<?> robot) {
    long nOfVoxel = robot.getVoxels().values().stream().filter(Objects::nonNull).count();
    return Collections.nCopies((int) nOfVoxel, 0d);
  }

}