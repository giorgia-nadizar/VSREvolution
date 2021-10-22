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

import com.google.common.base.Stopwatch;
import it.units.erallab.builder.DirectNumbersGrid;
import it.units.erallab.builder.FunctionGrid;
import it.units.erallab.builder.FunctionNumbersGrid;
import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.builder.evolver.*;
import it.units.erallab.builder.phenotype.FGraph;
import it.units.erallab.builder.phenotype.MLP;
import it.units.erallab.builder.phenotype.PruningMLP;
import it.units.erallab.builder.phenotype.RNN;
import it.units.erallab.builder.robot.*;
import it.units.erallab.hmsrobots.core.controllers.Controller;
import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.controllers.PruningMultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.tasks.periodiclocomotion.PeriodicLocomotion;
import it.units.erallab.hmsrobots.tasks.periodiclocomotion.PeriodicOutcome;
import it.units.erallab.hmsrobots.util.RobotUtils;
import it.units.malelab.jgea.Worker;
import it.units.malelab.jgea.core.Individual;
import it.units.malelab.jgea.core.evolver.Event;
import it.units.malelab.jgea.core.evolver.Evolver;
import it.units.malelab.jgea.core.evolver.stopcondition.FitnessEvaluations;
import it.units.malelab.jgea.core.listener.*;
import it.units.malelab.jgea.core.listener.telegram.TelegramUpdater;
import it.units.malelab.jgea.core.order.PartialComparator;
import it.units.malelab.jgea.core.util.Misc;
import it.units.malelab.jgea.core.util.Pair;
import it.units.malelab.jgea.core.util.TextPlotter;
import org.dyn4j.dynamics.Settings;

import java.io.File;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;
import java.util.stream.Collectors;

import static it.units.erallab.hmsrobots.util.Utils.params;
import static it.units.malelab.jgea.core.listener.NamedFunctions.*;
import static it.units.malelab.jgea.core.util.Args.*;

/**
 * @author eric
 */
public class PeriodicLocomotionEvolution extends Worker {

  private final static Settings PHYSICS_SETTINGS = new Settings();
  public static final String MAPPER_PIPE_CHAR = "<";

  public PeriodicLocomotionEvolution(String[] args) {
    super(args);
  }

  public static void main(String[] args) {
    new PeriodicLocomotionEvolution(args);
  }

  @Override
  public void run() {
    int spectrumSize = 10;
    double spectrumMinFreq = 0d;
    double spectrumMaxFreq = 5d;
    double episodeTime = d(a("episodeTime", "30"));
    double videoEpisodeTime = d(a("videoEpisodeTime", "10"));
    double videoEpisodeTransientTime = d(a("videoEpisodeTransientTime", "0"));
    List<Integer> terrainLengths = i(l(a("terrainLength", "80")));
    int nEvals = i(a("nEvals", "100"));
    int[] seeds = ri(a("seed", "0:1"));
    String experimentName = a("expName", "short");
    List<String> targetShapeNames = l(a("shape", "biped-4x3"));
    List<String> targetSensorConfigNames = l(a("sensorConfig", "spinedTouch-t-f-0.01"));
    List<String> transformationNames = l(a("transformation", "identity"));
    List<String> evolverNames = l(a("evolver", "ES-10-0.35"));
    List<String> mapperNames = l(a("mapper", "fixedCentralized<MLP-2-2-tanh"));
    String lastFileName = a("lastFile", null);
    String bestFileName = a("bestFile", null);
    String allFileName = a("allFile", null);
    boolean deferred = a("deferred", "true").startsWith("t");
    String telegramBotId = a("telegramBotId", null);
    long telegramChatId = Long.parseLong(a("telegramChatId", "0"));
    List<String> serializationFlags = l(a("serialization", "")); //last,best,all
    boolean output = a("output", "false").startsWith("t");
    Function<PeriodicOutcome, Double> fitnessFunction = PeriodicOutcome::getCoverage;
    //consumers
    List<NamedFunction<Event<?, ? extends Robot<?>, ? extends PeriodicOutcome>, ?>> keysFunctions = Utils.keysFunctions();
    List<NamedFunction<Event<?, ? extends Robot<?>, ? extends PeriodicOutcome>, ?>> basicFunctions = Utils.basicFunctions();
    List<NamedFunction<Individual<?, ? extends Robot<?>, ? extends PeriodicOutcome>, ?>> basicIndividualFunctions = Utils.individualFunctions(fitnessFunction);
    List<NamedFunction<Event<?, ? extends Robot<?>, ? extends PeriodicOutcome>, ?>> populationFunctions = Utils.populationFunctions(fitnessFunction);
    List<NamedFunction<Event<?, ? extends Robot<?>, ? extends PeriodicOutcome>, ?>> visualFunctions = Utils.visualFunctions(fitnessFunction);
    List<NamedFunction<PeriodicOutcome, ?>> basicOutcomeFunctions = Utils.basicOutcomeFunctions();
    List<NamedFunction<PeriodicOutcome, ?>> detailedOutcomeFunctions = Utils.detailedOutcomeFunctions(spectrumMinFreq, spectrumMaxFreq, spectrumSize);
    List<NamedFunction<PeriodicOutcome, ?>> visualOutcomeFunctions = Utils.visualOutcomeFunctions(spectrumMinFreq, spectrumMaxFreq);
    Listener.Factory<Event<?, ? extends Robot<?>, ? extends PeriodicOutcome>> factory = Listener.Factory.deaf();
    //screen listener
    if (bestFileName == null || output) {
      factory = factory.and(new TabularPrinter<>(Misc.concat(List.of(
          basicFunctions,
          populationFunctions,
          visualFunctions,
          NamedFunction.then(best(), basicIndividualFunctions),
          NamedFunction.then(as(PeriodicOutcome.class).of(fitness()).of(best()), basicOutcomeFunctions),
          NamedFunction.then(as(PeriodicOutcome.class).of(fitness()).of(best()), visualOutcomeFunctions)
      ))));
    }
    //file listeners
    if (lastFileName != null) {
      factory = factory.and(new CSVPrinter<>(Misc.concat(List.of(
          keysFunctions,
          basicFunctions,
          populationFunctions,
          NamedFunction.then(best(), basicIndividualFunctions),
          NamedFunction.then(as(PeriodicOutcome.class).of(fitness()).of(best()), basicOutcomeFunctions),
          NamedFunction.then(as(PeriodicOutcome.class).of(fitness()).of(best()), detailedOutcomeFunctions),
          NamedFunction.then(best(), Utils.serializationFunction(serializationFlags.contains("last")))
      )), new File(lastFileName)
      ).onLast());
    }
    if (bestFileName != null) {
      factory = factory.and(new CSVPrinter<>(Misc.concat(List.of(
          keysFunctions,
          basicFunctions,
          populationFunctions,
          NamedFunction.then(best(), basicIndividualFunctions),
          NamedFunction.then(as(PeriodicOutcome.class).of(fitness()).of(best()), basicOutcomeFunctions),
          NamedFunction.then(as(PeriodicOutcome.class).of(fitness()).of(best()), detailedOutcomeFunctions),
          NamedFunction.then(best(), Utils.serializationFunction(serializationFlags.contains("best")))
      )), new File(bestFileName)
      ));
    }
    if (allFileName != null) {
      factory = factory.and(Listener.Factory.forEach(
          event -> event.getOrderedPopulation().all().stream()
              .map(i -> Pair.of(event, i))
              .collect(Collectors.toList()),
          new CSVPrinter<>(
              Misc.concat(List.of(
                  NamedFunction.then(f("event", Pair::first), keysFunctions),
                  NamedFunction.then(f("event", Pair::first), basicFunctions),
                  NamedFunction.then(f("individual", Pair::second), basicIndividualFunctions),
                  NamedFunction.then(f("individual", Pair::second), Utils.serializationFunction(serializationFlags.contains("all")))
              )),
              new File(allFileName)
          )
      ));
    }
    if (telegramBotId != null && telegramChatId != 0) {
      factory = factory.and(new TelegramUpdater<>(List.of(
          Utils.lastEventToString(fitnessFunction),
          Utils.fitnessPlot(fitnessFunction),
          Utils.centerPositionPlot(),
          Utils.bestVideo(videoEpisodeTransientTime, videoEpisodeTime, PHYSICS_SETTINGS)
      ), telegramBotId, telegramChatId));
    }
    //summarize params
    L.info("Experiment name: " + experimentName);
    L.info("Evolvers: " + evolverNames);
    L.info("Mappers: " + mapperNames);
    L.info("Shapes: " + targetShapeNames);
    L.info("Sensor configs: " + targetSensorConfigNames);
    L.info("Terrain lengths: " + terrainLengths);
    //start iterations
    int nOfRuns = seeds.length * terrainLengths.size() * targetShapeNames.size() * targetSensorConfigNames.size() * mapperNames.size() * transformationNames.size() * evolverNames.size();
    int counter = 0;
    for (int seed : seeds) {
      for (int terrainLength : terrainLengths) {
        for (String targetShapeName : targetShapeNames) {
          for (String targetSensorConfigName : targetSensorConfigNames) {
            for (String mapperName : mapperNames) {
              for (String transformationName : transformationNames) {
                for (String evolverName : evolverNames) {
                  counter = counter + 1;
                  final Random random = new Random(seed);
                  //prepare keys
                  Map<String, Object> keys = Map.ofEntries(
                      Map.entry("experiment.name", experimentName),
                      Map.entry("seed", seed),
                      Map.entry("terrain.length", terrainLength),
                      Map.entry("shape", targetShapeName),
                      Map.entry("sensor.config", targetSensorConfigName),
                      Map.entry("mapper", mapperName),
                      Map.entry("transformation", transformationName),
                      Map.entry("evolver", evolverName)
                  );
                  Robot<?> target = new Robot<>(
                      Controller.empty(),
                      RobotUtils.buildSensorizingFunction(targetSensorConfigName).apply(RobotUtils.buildShape(targetShapeName))
                  );
                  //build evolver
                  Evolver<?, Robot<?>, PeriodicOutcome> evolver;
                  try {
                    evolver = buildEvolver(evolverName, mapperName, target, fitnessFunction);
                  } catch (ClassCastException | IllegalArgumentException e) {
                    L.warning(String.format(
                        "Cannot instantiate %s for %s: %s",
                        evolverName,
                        mapperName,
                        e
                    ));
                    continue;
                  }
                  Listener<Event<?, ? extends Robot<?>, ? extends PeriodicOutcome>> listener = Listener.all(List.of(
                      new EventAugmenter(keys),
                      factory.build()
                  ));
                  if (deferred) {
                    listener = listener.deferred(executorService);
                  }
                  //optimize
                  Stopwatch stopwatch = Stopwatch.createStarted();
                  L.info(String.format("Progress %s (%d/%d); Starting %s",
                      TextPlotter.horizontalBar(counter - 1, 0, nOfRuns, 8),
                      counter, nOfRuns,
                      keys
                  ));
                  //build task
                  try {
                    Collection<Robot<?>> solutions = evolver.solve(
                        buildPeriodicLocomotionTask(terrainLength, episodeTime),
                        new FitnessEvaluations(nEvals),
                        random,
                        executorService,
                        listener
                    );
                    L.info(String.format("Progress %s (%d/%d); Done: %d solutions in %4ds",
                        TextPlotter.horizontalBar(counter, 0, nOfRuns, 8),
                        counter, nOfRuns,
                        solutions.size(),
                        stopwatch.elapsed(TimeUnit.SECONDS)
                    ));
                  } catch (Exception e) {
                    L.severe(String.format("Cannot complete %s due to %s",
                        keys,
                        e
                    ));
                    e.printStackTrace(); // TODO possibly to be removed
                  }
                }
              }
            }
          }
        }
      }
    }
    factory.shutdown();
  }

  private static EvolverBuilder<?> getEvolverBuilderFromName(String name) {
    String numGA = "numGA-(?<nPop>\\d+)-(?<diversity>(t|f))";
    String numGASpeciated = "numGASpec-(?<nPop>\\d+)-(?<nSpecies>\\d+)-(?<criterion>(" + Arrays.stream(DoublesSpeciated.SpeciationCriterion.values()).map(c -> c.name().toLowerCase(Locale.ROOT)).collect(Collectors.joining("|")) + "))";
    String cmaES = "CMAES";
    String eS = "ES-(?<nPop>\\d+)-(?<sigma>\\d+(\\.\\d+)?)";
    Map<String, String> params;
    if ((params = params(numGA, name)) != null) {
      return new DoublesStandard(
          Integer.parseInt(params.get("nPop")),
          (int) Math.max(Math.round((double) Integer.parseInt(params.get("nPop")) / 10d), 3),
          0.75d,
          params.get("diversity").equals("t")
      );
    }
    if ((params = params(numGASpeciated, name)) != null) {
      return new DoublesSpeciated(
          Integer.parseInt(params.get("nPop")),
          Integer.parseInt(params.get("nSpecies")),
          0.75d,
          DoublesSpeciated.SpeciationCriterion.valueOf(params.get("criterion").toUpperCase())
      );
    }
    if ((params = params(eS, name)) != null) {
      return new ES(
          Double.parseDouble(params.get("sigma")),
          Integer.parseInt(params.get("nPop"))
      );
    }
    if ((params = params(cmaES, name)) != null) {
      return new CMAES();
    }
    throw new IllegalArgumentException(String.format("Unknown evolver builder name: %s", name));
  }

  @SuppressWarnings({"unchecked", "rawtypes"})
  private static PrototypedFunctionBuilder<?, ?> getMapperBuilderFromName(String name) {
    String fixedCentralized = "fixedCentralized";
    String fixedHomoDistributed = "fixedHomoDist-(?<nSignals>\\d+)";
    String fixedHeteroDistributed = "fixedHeteroDist-(?<nSignals>\\d+)";
    String fixedPhasesFunction = "fixedPhasesFunct-(?<f>\\d+)";
    String fixedPhases = "fixedPhases-(?<f>\\d+)";
    String bodySin = "bodySin-(?<fullness>\\d+(\\.\\d+)?)-(?<minF>\\d+(\\.\\d+)?)-(?<maxF>\\d+(\\.\\d+)?)";
    String bodyAndHomoDistributed = "bodyAndHomoDist-(?<fullness>\\d+(\\.\\d+)?)-(?<nSignals>\\d+)-(?<nLayers>\\d+)";
    String sensorAndBodyAndHomoDistributed = "sensorAndBodyAndHomoDist-(?<fullness>\\d+(\\.\\d+)?)-(?<nSignals>\\d+)-(?<nLayers>\\d+)-(?<position>(t|f))";
    String sensorCentralized = "sensorCentralized-(?<nLayers>\\d+)";
    String mlp = "MLP-(?<ratio>\\d+(\\.\\d+)?)-(?<nLayers>\\d+)(-(?<actFun>(sin|tanh|sigmoid|relu)))?";
    String pruningMlp = "pMLP-(?<ratio>\\d+(\\.\\d+)?)-(?<nLayers>\\d+)-(?<actFun>(sin|tanh|sigmoid|relu))-(?<pruningTime>\\d+(\\.\\d+)?)-(?<pruningRate>0(\\.\\d+)?)-(?<criterion>(weight|abs_signal_mean|random))";
    String rnn = "RNN-(?<recurrentNeurons>\\d+)";
    String directNumGrid = "directNumGrid";
    String functionNumGrid = "functionNumGrid";
    String fgraph = "fGraph";
    String functionGrid = "fGrid-(?<innerMapper>.*)";
    Map<String, String> params;
    //robot mappers
    if ((params = params(fixedCentralized, name)) != null) {
      return new FixedCentralized();
    }
    if ((params = params(fixedHomoDistributed, name)) != null) {
      return new FixedHomoDistributed(
          Integer.parseInt(params.get("nSignals"))
      );
    }
    if ((params = params(fixedHeteroDistributed, name)) != null) {
      return new FixedHeteroDistributed(
          Integer.parseInt(params.get("nSignals"))
      );
    }
    if ((params = params(fixedPhasesFunction, name)) != null) {
      return new FixedPhaseFunction(
          Double.parseDouble(params.get("f")),
          1d
      );
    }
    if ((params = params(fixedPhases, name)) != null) {
      return new FixedPhaseValues(
          Double.parseDouble(params.get("f")),
          1d
      );
    }
    if ((params = params(bodyAndHomoDistributed, name)) != null) {
      return new BodyAndHomoDistributed(
          Integer.parseInt(params.get("nSignals")),
          Double.parseDouble(params.get("fullness"))
      )
          .compose(PrototypedFunctionBuilder.of(List.of(
              new MLP(2d, 3, MultiLayerPerceptron.ActivationFunction.SIN),
              new MLP(0.65d, Integer.parseInt(params.get("nLayers")))
          )))
          .compose(PrototypedFunctionBuilder.merger());
    }
    if ((params = params(sensorAndBodyAndHomoDistributed, name)) != null) {
      return new SensorAndBodyAndHomoDistributed(
          Integer.parseInt(params.get("nSignals")),
          Double.parseDouble(params.get("fullness")),
          params.get("position").equals("t")
      )
          .compose(PrototypedFunctionBuilder.of(List.of(
              new MLP(2d, 3, MultiLayerPerceptron.ActivationFunction.SIN),
              new MLP(1.5d, Integer.parseInt(params.get("nLayers")))
          )))
          .compose(PrototypedFunctionBuilder.merger());
    }
    if ((params = params(bodySin, name)) != null) {
      return new BodyAndSinusoidal(
          Double.parseDouble(params.get("minF")),
          Double.parseDouble(params.get("maxF")),
          Double.parseDouble(params.get("fullness")),
          Set.of(BodyAndSinusoidal.Component.FREQUENCY, BodyAndSinusoidal.Component.PHASE, BodyAndSinusoidal.Component.AMPLITUDE)
      );
    }
    if ((params = params(fixedHomoDistributed, name)) != null) {
      return new FixedHomoDistributed(
          Integer.parseInt(params.get("nSignals"))
      );
    }
    if ((params = params(sensorCentralized, name)) != null) {
      return new SensorCentralized()
          .compose(PrototypedFunctionBuilder.of(List.of(
              new MLP(2d, 3, MultiLayerPerceptron.ActivationFunction.SIN),
              new MLP(1.5d, Integer.parseInt(params.get("nLayers")))
          )))
          .compose(PrototypedFunctionBuilder.merger());
    }
    //function mappers
    if ((params = params(mlp, name)) != null) {
      return new MLP(
          Double.parseDouble(params.get("ratio")),
          Integer.parseInt(params.get("nLayers")),
          params.containsKey("actFun") ? MultiLayerPerceptron.ActivationFunction.valueOf(params.get("actFun").toUpperCase()) : MultiLayerPerceptron.ActivationFunction.TANH
      );
    }
    if ((params = params(pruningMlp, name)) != null) {
      return new PruningMLP(
          Double.parseDouble(params.get("ratio")),
          Integer.parseInt(params.get("nLayers")),
          MultiLayerPerceptron.ActivationFunction.valueOf(params.get("actFun").toUpperCase()),
          Double.parseDouble(params.get("pruningTime")),
          Double.parseDouble(params.get("pruningRate")),
          PruningMultiLayerPerceptron.Context.NETWORK,
          PruningMultiLayerPerceptron.Criterion.valueOf(params.get("criterion").toUpperCase())

      );
    }
    if ((params = params(rnn, name)) != null) {
      return new RNN(
          Integer.parseInt(params.get("recurrentNeurons"))
      );
    }
    if ((params = params(fgraph, name)) != null) {
      return new FGraph();
    }
    //misc
    if ((params = params(functionGrid, name)) != null) {
      return new FunctionGrid((PrototypedFunctionBuilder) getMapperBuilderFromName(params.get("innerMapper")));
    }
    if ((params = params(directNumGrid, name)) != null) {
      return new DirectNumbersGrid();
    }
    if ((params = params(functionNumGrid, name)) != null) {
      return new FunctionNumbersGrid();
    }
    throw new IllegalArgumentException(String.format("Unknown mapper name: %s", name));
  }

  @SuppressWarnings({"unchecked", "rawtypes"})
  private static Evolver<?, Robot<?>, PeriodicOutcome> buildEvolver(String evolverName, String robotMapperName, Robot<?>
      target, Function<PeriodicOutcome, Double> outcomeMeasure) {
    PrototypedFunctionBuilder<?, ?> mapperBuilder = null;
    for (String piece : robotMapperName.split(MAPPER_PIPE_CHAR)) {
      if (mapperBuilder == null) {
        mapperBuilder = getMapperBuilderFromName(piece);
      } else {
        mapperBuilder = mapperBuilder.compose((PrototypedFunctionBuilder) getMapperBuilderFromName(piece));
      }
    }
    return getEvolverBuilderFromName(evolverName).build(
        (PrototypedFunctionBuilder) mapperBuilder,
        target,
        PartialComparator.from(Double.class).comparing(outcomeMeasure).reversed()
    );
  }

  private static Function<Robot<?>, PeriodicOutcome> buildPeriodicLocomotionTask(int terrainLength, double episodeT) {
    return r -> new PeriodicLocomotion(
        episodeT,
        terrainLength,
        PHYSICS_SETTINGS
    ).apply(r);
  }

}
