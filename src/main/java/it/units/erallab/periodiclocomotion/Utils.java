/*
 * Copyright 2020 Eric Medvet <eric.medvet@gmail.com> (as eric)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package it.units.erallab.periodiclocomotion;

import it.units.erallab.hmsrobots.behavior.BehaviorUtils;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.snapshots.VoxelPoly;
import it.units.erallab.hmsrobots.tasks.locomotion.Outcome;
import it.units.erallab.hmsrobots.tasks.periodiclocomotion.PeriodicLocomotion;
import it.units.erallab.hmsrobots.tasks.periodiclocomotion.PeriodicOutcome;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.erallab.hmsrobots.viewers.GridFileWriter;
import it.units.erallab.hmsrobots.viewers.VideoUtils;
import it.units.malelab.jgea.core.Individual;
import it.units.malelab.jgea.core.evolver.Event;
import it.units.malelab.jgea.core.listener.Accumulator;
import it.units.malelab.jgea.core.listener.NamedFunction;
import it.units.malelab.jgea.core.listener.NamedFunctions;
import it.units.malelab.jgea.core.listener.TableBuilder;
import it.units.malelab.jgea.core.util.*;
import org.dyn4j.dynamics.Settings;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.function.Function;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static it.units.malelab.jgea.core.listener.NamedFunctions.*;

/**
 * @author eric
 */
public class Utils {

  private static final Logger L = Logger.getLogger(Utils.class.getName());

  private Utils() {
  }

  public static List<NamedFunction<Event<?, ? extends Robot<?>, ? extends PeriodicOutcome>, ?>> keysFunctions() {
    return List.of(
        eventAttribute("experiment.name"),
        eventAttribute("seed", "%2d"),
        eventAttribute("terrain.length"),
        eventAttribute("shape"),
        eventAttribute("sensor.config"),
        eventAttribute("mapper"),
        eventAttribute("transformation"),
        eventAttribute("evolver")
    );
  }

  public static List<NamedFunction<Event<?, ? extends Robot<?>, ? extends PeriodicOutcome>, ?>> basicFunctions() {
    return List.of(
        iterations(),
        births(),
        fitnessEvaluations(),
        elapsedSeconds()
    );
  }

  public static List<NamedFunction<Individual<?, ? extends Robot<?>, ? extends PeriodicOutcome>, ?>> serializationFunction(boolean flag) {
    if (!flag) {
      return List.of();
    }
    return List.of(f("serialized", r -> SerializationUtils.serialize(r, SerializationUtils.Mode.GZIPPED_JSON)).of(solution()));
  }

  public static List<NamedFunction<Individual<?, ? extends Robot<?>, ? extends PeriodicOutcome>, ?>> individualFunctions(Function<PeriodicOutcome, Double> fitnessFunction) {
    NamedFunction<Individual<?, ? extends Robot<?>, ? extends PeriodicOutcome>, ?> size = size().of(genotype());
    return List.of(
        f("w", "%2d", (Function<Grid<?>, Number>) Grid::getW)
            .of(f("shape", (Function<Robot<?>, Grid<?>>) Robot::getVoxels))
            .of(solution()),
        f("h", "%2d", (Function<Grid<?>, Number>) Grid::getH)
            .of(f("shape", (Function<Robot<?>, Grid<?>>) Robot::getVoxels))
            .of(solution()),
        f("num.voxel", "%2d", (Function<Grid<?>, Number>) g -> g.count(Objects::nonNull))
            .of(f("shape", (Function<Robot<?>, Grid<?>>) Robot::getVoxels))
            .of(solution()),
        size.reformat("%5d"),
        genotypeBirthIteration(),
        f("fitness", "%5.3f", fitnessFunction).of(fitness())
    );
  }

  public static List<NamedFunction<Event<?, ? extends Robot<?>, ? extends PeriodicOutcome>, ?>> populationFunctions(Function<PeriodicOutcome, Double> fitnessFunction) {
    NamedFunction<Event<?, ? extends Robot<?>, ? extends PeriodicOutcome>, ?> min = min(Double::compare).of(each(f("fitness", fitnessFunction).of(fitness()))).of(all());
    NamedFunction<Event<?, ? extends Robot<?>, ? extends PeriodicOutcome>, ?> median = median(Double::compare).of(each(f("fitness", fitnessFunction).of(fitness()))).of(all());
    return List.of(
        size().of(all()),
        size().of(firsts()),
        size().of(lasts()),
        uniqueness().of(each(genotype())).of(all()),
        uniqueness().of(each(solution())).of(all()),
        uniqueness().of(each(fitness())).of(all()),
        min.reformat("%+4.3f"),
        median.reformat("%5.3f")
    );
  }

  public static List<NamedFunction<Event<?, ? extends Robot<?>, ? extends PeriodicOutcome>, ?>> visualFunctions(Function<PeriodicOutcome, Double> fitnessFunction) {
    return List.of(
        hist(8)
            .of(each(f("fitness", fitnessFunction).of(fitness())))
            .of(all()),
        hist(8)
            .of(each(f("num.voxels", (Function<Grid<?>, Number>) g -> g.count(Objects::nonNull))
                .of(f("shape", (Function<Robot<?>, Grid<?>>) Robot::getVoxels))
                .of(solution())))
            .of(all()),
        f("minimap", "%4s", (Function<Grid<?>, String>) g -> TextPlotter.binaryMap(
            g.toArray(Objects::nonNull),
            (int) Math.min(Math.ceil((float) g.getW() / (float) g.getH() * 2f), 4)))
            .of(f("shape", (Function<Robot<?>, Grid<?>>) Robot::getVoxels))
            .of(solution()).of(best()),
        f("average.posture.minimap", "%2s", (Function<PeriodicOutcome, String>) o -> TextPlotter.binaryMap(o.getAveragePosture(8).toArray(b -> b), 2))
            .of(fitness()).of(best())
    );
  }

  public static List<NamedFunction<PeriodicOutcome, ?>> basicOutcomeFunctions() {
    return List.of(
        f("computation.time", "%4.2f", PeriodicOutcome::getComputationTime),
        f("distance", "%5.1f", PeriodicOutcome::getDistance),
        f("velocity", "%5.1f", PeriodicOutcome::getVelocity),
        f("corrected.efficiency", "%5.2f", PeriodicOutcome::getCorrectedEfficiency),
        f("area.ratio.power", "%5.1f", PeriodicOutcome::getAreaRatioPower),
        f("control.power", "%5.1f", PeriodicOutcome::getControlPower)
    );
  }

  public static List<NamedFunction<PeriodicOutcome, ?>> detailedOutcomeFunctions(double spectrumMinFreq, double spectrumMaxFreq, int spectrumSize) {
    return Misc.concat(List.of(
        NamedFunction.then(cachedF(
            "center.x.spectrum",
            (PeriodicOutcome o) -> new ArrayList<>(o.getCenterXVelocitySpectrum(spectrumMinFreq, spectrumMaxFreq, spectrumSize).values())
            ),
            IntStream.range(0, spectrumSize).mapToObj(NamedFunctions::nth).collect(Collectors.toList())
        ),
        NamedFunction.then(cachedF(
            "center.y.spectrum",
            (PeriodicOutcome o) -> new ArrayList<>(o.getCenterYVelocitySpectrum(spectrumMinFreq, spectrumMaxFreq, spectrumSize).values())
            ),
            IntStream.range(0, spectrumSize).mapToObj(NamedFunctions::nth).collect(Collectors.toList())
        ),
        NamedFunction.then(cachedF(
            "center.angle.spectrum",
            (PeriodicOutcome o) -> new ArrayList<>(o.getCenterAngleSpectrum(spectrumMinFreq, spectrumMaxFreq, spectrumSize).values())
            ),
            IntStream.range(0, spectrumSize).mapToObj(NamedFunctions::nth).collect(Collectors.toList())
        ),
        NamedFunction.then(cachedF(
            "footprints.spectra",
            (PeriodicOutcome o) -> o.getFootprintsSpectra(4, spectrumMinFreq, spectrumMaxFreq, spectrumSize).stream()
                .map(SortedMap::values)
                .flatMap(Collection::stream)
                .collect(Collectors.toList())
            ),
            IntStream.range(0, 4 * spectrumSize).mapToObj(NamedFunctions::nth).collect(Collectors.toList())
        )
    ));
  }

  public static List<NamedFunction<PeriodicOutcome, ?>> visualOutcomeFunctions(double spectrumMinFreq, double spectrumMaxFreq) {
    return Misc.concat(List.of(
        List.of(
            cachedF("center.x.spectrum", "%4.4s", o -> TextPlotter.barplot(
                new ArrayList<>(o.getCenterXVelocitySpectrum(spectrumMinFreq, spectrumMaxFreq, 4).values())
            )),
            cachedF("center.y.spectrum", "%4.4s", o -> TextPlotter.barplot(
                new ArrayList<>(o.getCenterYVelocitySpectrum(spectrumMinFreq, spectrumMaxFreq, 4).values())
            )),
            cachedF("center.angle.spectrum", "%4.4s", o -> TextPlotter.barplot(
                new ArrayList<>(o.getCenterAngleSpectrum(spectrumMinFreq, spectrumMaxFreq, 4).values())
            ))
        ),
        NamedFunction.then(cachedF("footprints", o -> o.getFootprintsSpectra(3, spectrumMinFreq, spectrumMaxFreq, 4)),
            List.of(
                cachedF("left.spectrum", "%4.4s", l -> TextPlotter.barplot(new ArrayList<>(l.get(0).values()))),
                cachedF("center.spectrum", "%4.4s", l -> TextPlotter.barplot(new ArrayList<>(l.get(1).values()))),
                cachedF("right.spectrum", "%4.4s", l -> TextPlotter.barplot(new ArrayList<>(l.get(2).values())))
            )
        )
    ));
  }

  public static Accumulator.Factory<Event<?, ? extends Robot<?>, ? extends PeriodicOutcome>, String> lastEventToString(Function<PeriodicOutcome, Double> fitnessFunction) {
    final List<NamedFunction<Event<?, ? extends Robot<?>, ? extends PeriodicOutcome>, ?>> functions = Misc.concat(List.of(
        keysFunctions(),
        basicFunctions(),
        populationFunctions(fitnessFunction),
        NamedFunction.then(best(), individualFunctions(fitnessFunction)),
        NamedFunction.then(as(PeriodicOutcome.class).of(fitness()).of(best()), basicOutcomeFunctions())
    ));
    return Accumulator.Factory.<Event<?, ? extends Robot<?>, ? extends PeriodicOutcome>>last().then(
        e -> functions.stream()
            .map(f -> f.getName() + ": " + f.applyAndFormat(e))
            .collect(Collectors.joining("\n"))
    );
  }

  public static Accumulator.Factory<Event<?, ? extends Robot<?>, ? extends PeriodicOutcome>, BufferedImage> fitnessPlot(Function<PeriodicOutcome, Double> fitnessFunction) {
    return new TableBuilder<Event<?, ? extends Robot<?>, ? extends PeriodicOutcome>, Number>(List.of(
        iterations(),
        f("fitness", fitnessFunction).of(fitness()).of(best()),
        min(Double::compare).of(each(f("fitness", fitnessFunction).of(fitness()))).of(all()),
        median(Double::compare).of(each(f("fitness", fitnessFunction).of(fitness()))).of(all())
    )).then(ImagePlotters.xyLines(600, 400));
  }

  public static Accumulator.Factory<Event<?, ? extends Robot<?>, ? extends PeriodicOutcome>, BufferedImage> centerPositionPlot() {
    return Accumulator.Factory.<Event<?, ? extends Robot<?>, ? extends PeriodicOutcome>>last().then(
        event -> {
          Outcome o = Misc.first(event.getOrderedPopulation().firsts()).getFitness();
          Table<Number> table = new ArrayTable<>(List.of("x", "y", "terrain.y"));
          o.getObservations().values().forEach(obs -> {
            VoxelPoly poly = BehaviorUtils.getCentralElement(obs.getVoxelPolies());
            table.addRow(List.of(
                poly.center().x,
                poly.center().y,
                obs.getTerrainHeight()
            ));
          });
          return table;
        }
    )
        .then(ImagePlotters.xyLines(600, 400));
  }

  public static Accumulator.Factory<Event<?, ? extends Robot<?>, ? extends PeriodicOutcome>, File> bestVideo(double transientTime, double episodeTime, Settings settings) {
    return Accumulator.Factory.<Event<?, ? extends Robot<?>, ? extends PeriodicOutcome>>last().then(
        event -> {
          int terrainLength = Integer.parseInt(event.getAttributes().get("terrain.length").toString());
          Robot<?> robot = SerializationUtils.clone(Misc.first(event.getOrderedPopulation().firsts()).getSolution());
          PeriodicLocomotion locomotion = new PeriodicLocomotion(
              episodeTime,
              terrainLength,
              settings
          );
          File file;
          try {
            file = File.createTempFile("robot-video", ".mp4");
            GridFileWriter.save(locomotion, robot, 300, 200, transientTime, 25, VideoUtils.EncoderFacility.JCODEC, file);
            file.deleteOnExit();
          } catch (IOException ioException) {
            L.warning(String.format("Cannot save video of best: %s", ioException));
            return null;
          }
          return file;
        }
    );
  }

}
