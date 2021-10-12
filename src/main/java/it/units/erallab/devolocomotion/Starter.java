package it.units.erallab.devolocomotion;

import com.google.common.base.Stopwatch;
import it.units.erallab.builder.DirectNumbersGrid;
import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.builder.devofunction.DevoHomoMLP;
import it.units.erallab.builder.devofunction.DevoPhasesValues;
import it.units.erallab.hmsrobots.core.controllers.Controller;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.tasks.devolocomotion.DevoLocomotion;
import it.units.erallab.hmsrobots.tasks.devolocomotion.DevoOutcome;
import it.units.erallab.hmsrobots.tasks.locomotion.Locomotion;
import it.units.erallab.hmsrobots.tasks.locomotion.Outcome;
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
import it.units.malelab.jgea.core.util.TextPlotter;

import java.io.File;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;
import java.util.function.UnaryOperator;
import java.util.stream.Collectors;

import static it.units.erallab.devolocomotion.NamedFunctions.*;
import static it.units.erallab.hmsrobots.util.Utils.params;
import static it.units.malelab.jgea.core.listener.NamedFunctions.best;
import static it.units.malelab.jgea.core.listener.NamedFunctions.f;
import static it.units.malelab.jgea.core.util.Args.*;

/**
 * @author "Eric Medvet" on 2021/09/27 for VSREvolution
 */
public class Starter extends Worker {

  public Starter(String[] args) {
    super(args);
  }

  public static void main(String[] args) {
    new Starter(args);
  }

  public static class DevoValidationOutcome {
    private final Event<?, ? extends UnaryOperator<Robot<?>>, ? extends DevoOutcome> event;
    private final Map<String, Object> keys;
    private final DevoOutcome devoOutcome;

    public DevoValidationOutcome(Event<?, ? extends UnaryOperator<Robot<?>>, ? extends DevoOutcome> event, Map<String, Object> keys, DevoOutcome devoOutcome) {
      this.event = event;
      this.keys = keys;
      this.devoOutcome = devoOutcome;
    }
  }

  @Override
  public void run() {
    //main params
    double episodeTime = d(a("episodeTime", "60"));
    double stageMaxTime = d(a("stageMaxTime", "20"));
    double stageMinDistance = d(a("stageMinDistance", "20"));
    double validationEpisodeTime = d(a("validationEpisodeTime", Double.toString(episodeTime)));
    double validationStageMaxTime = d(a("validationStageMaxTime", Double.toString(stageMaxTime)));
    double validationStageMinDistance = d(a("validationStageMinDistance", Double.toString(stageMinDistance)));
    int nEvals = i(a("nEvals", "800"));
    int[] seeds = ri(a("seed", "0:1"));
    String experimentName = a("expName", "short");
    int gridW = i(a("gridW", "5"));
    int gridH = i(a("gridH", "5"));
    List<String> terrainNames = l(a("terrain", "flat"));
    List<String> devoFunctionNames = l(a("devoFunction", "devoPhases-1.0-5-1<directNumGrid"));
    List<String> targetSensorConfigNames = l(a("sensorConfig", "uniform-t+a-0.01"));
    List<String> evolverNames = l(a("evolver", "ES-16-0.35"));
    String lastFileName = a("lastFile", null);
    String bestFileName = a("bestFile", null);
    String validationFileName = a("validationFile", null);
    boolean deferred = a("deferred", "true").startsWith("t");
    List<String> serializationFlags = l(a("serialization", "")); //last,best,validation
    boolean output = a("output", "false").startsWith("t");
    String telegramBotId = a("telegramBotId", null);
    long telegramChatId = Long.parseLong(a("telegramChatId", "0"));
    List<String> validationTerrainNames = l(a("validationTerrain", "flat,downhill-30")).stream().filter(s -> !s.isEmpty()).collect(Collectors.toList());
    //fitness function
    Function<DevoOutcome, Double> fitnessFunction = devoOutcome -> devoOutcome.getLocomotionOutcomes().stream().mapToDouble(Outcome::getDistance).sum();
    //consumers
    List<NamedFunction<Event<?, ? extends UnaryOperator<Robot<?>>, ? extends DevoOutcome>, ?>> keysFunctions = keysFunctions();
    List<NamedFunction<Event<?, ? extends UnaryOperator<Robot<?>>, ? extends DevoOutcome>, ?>> basicFunctions = basicFunctions();
    List<NamedFunction<Event<?, ? extends UnaryOperator<Robot<?>>, ? extends DevoOutcome>, ?>> populationFunctions = populationFunctions(fitnessFunction);
    List<NamedFunction<Individual<?, ? extends UnaryOperator<Robot<?>>, ? extends DevoOutcome>, ?>> basicIndividualFunctions = basicIndividualFunctions(fitnessFunction);
    List<NamedFunction<DevoOutcome, ?>> outcomesFunctions = outcomesFunctions();
    List<NamedFunction<DevoOutcome, ?>> lastOutcomesFunctions = outcomesFunctions();
    List<NamedFunction<DevoOutcome, ?>> validationOutcomesFunctions = outcomesFunctions();
    if (serializationFlags.contains("best")) {
      List<NamedFunction<DevoOutcome, ?>> tempFunctions = new ArrayList<>(outcomesFunctions);
      tempFunctions.add(serializedDevoRobotsFunction());
      outcomesFunctions = tempFunctions;
    }
    if (serializationFlags.contains("validation")) {
      List<NamedFunction<DevoOutcome, ?>> tempFunctions = new ArrayList<>(validationOutcomesFunctions);
      tempFunctions.add(serializedDevoRobotsFunction());
      validationOutcomesFunctions = tempFunctions;
    }
    if (serializationFlags.contains("last")) {
      List<NamedFunction<DevoOutcome, ?>> tempFunctions = new ArrayList<>(lastOutcomesFunctions);
      tempFunctions.add(serializedDevoRobotsFunction());
      lastOutcomesFunctions = tempFunctions;
    }
    List<NamedFunction<Individual<?, ? extends UnaryOperator<Robot<?>>, ? extends DevoOutcome>, ?>> visualIndividualFunctions = visualIndividualFunctions();
    Listener.Factory<Event<?, ? extends UnaryOperator<Robot<?>>, ? extends DevoOutcome>> factory = Listener.Factory.deaf();
    NamedFunction<Event<?, ? extends UnaryOperator<Robot<?>>, ? extends DevoOutcome>, DevoOutcome> bestFitness = f("best.fitness", event -> Misc.first(event.getOrderedPopulation().firsts()).getFitness());
    //screen listener
    if ((bestFileName == null) || output) {
      factory = factory.and(new TabularPrinter<>(Misc.concat(List.of(
          basicFunctions,
          populationFunctions,
          NamedFunction.then(best(), basicIndividualFunctions),
          NamedFunction.then(best(), visualIndividualFunctions),
          NamedFunction.then(bestFitness, outcomesFunctions)
      ))));
    }
    //file listeners
    if (lastFileName != null) {
      factory = factory.and(new CSVPrinter<>(Misc.concat(List.of(
          keysFunctions,
          basicFunctions,
          populationFunctions,
          NamedFunction.then(best(), basicIndividualFunctions),
          NamedFunction.then(bestFitness, lastOutcomesFunctions)
      )), new File(lastFileName)
      ).onLast());
    }
    if (bestFileName != null) {
      factory = factory.and(new CSVPrinter<>(Misc.concat(List.of(
          keysFunctions,
          basicFunctions,
          populationFunctions,
          NamedFunction.then(best(), basicIndividualFunctions),
          NamedFunction.then(bestFitness, outcomesFunctions)
      )), new File(bestFileName)
      ));
    }
    //validation listener
    if (validationFileName != null) {
      if (validationTerrainNames.isEmpty()) {
        validationTerrainNames.add(terrainNames.get(0));
      }
      Listener.Factory<Event<?, ? extends UnaryOperator<Robot<?>>, ? extends DevoOutcome>> validationFactory = Listener.Factory.forEach(
          validation(validationTerrainNames, List.of(0), validationStageMinDistance, validationStageMaxTime, validationEpisodeTime),
          new CSVPrinter<>(
              Misc.concat(List.of(
                  NamedFunction.then(f("event", (DevoValidationOutcome vo) -> vo.event), keysFunctions),
                  NamedFunction.then(f("event", (DevoValidationOutcome vo) -> vo.event), basicFunctions),
                  NamedFunction.then(f("keys", (DevoValidationOutcome vo) -> vo.keys), List.of(
                      f("validation.terrain", (Map<String, Object> map) -> map.get("validation.terrain")),
                      f("validation.seed", "%2d", (Map<String, Object> map) -> map.get("validation.seed"))
                  )),
                  NamedFunction.then(f("outcome", (DevoValidationOutcome vo) -> vo.devoOutcome), validationOutcomesFunctions))
              ),
              new File(validationFileName)
          )
      ).onLast();
      factory = factory.and(validationFactory);
    }
    //telegram listener
    if (telegramBotId != null && telegramChatId != 0) {
      factory = factory.and(new TelegramUpdater<>(List.of(
          lastEventToString(fitnessFunction),
          fitnessPlot(fitnessFunction),
          bestVideo(stageMinDistance, stageMaxTime, episodeTime)
      ), telegramBotId, telegramChatId));
    }
    //summarize params
    L.info("Experiment name: " + experimentName);
    L.info("Evolvers: " + evolverNames);
    L.info("Devo functions: " + devoFunctionNames);
    L.info("Sensor configs: " + targetSensorConfigNames);
    L.info("Terrains: " + terrainNames);
    //start iterations
    int nOfRuns = seeds.length * terrainNames.size() * devoFunctionNames.size() * evolverNames.size();
    int counter = 0;
    for (int seed : seeds) {
      for (String terrainName : terrainNames) {
        for (String devoFunctionName : devoFunctionNames) {
          for (String targetSensorConfigName : targetSensorConfigNames) {
            for (String evolverName : evolverNames) {
              counter = counter + 1;
              final Random random = new Random(seed);
              //prepare keys
              Map<String, Object> keys = Map.ofEntries(
                  Map.entry("experiment.name", experimentName),
                  Map.entry("seed", seed),
                  Map.entry("terrain", terrainName),
                  Map.entry("devo.function", devoFunctionName),
                  Map.entry("sensor.config", targetSensorConfigName),
                  Map.entry("evolver", evolverName),
                  Map.entry("episode.time", episodeTime),
                  Map.entry("stage.max.time", stageMaxTime),
                  Map.entry("stage.min.dist", stageMinDistance)
              );
              //prepare target
              UnaryOperator<Robot<?>> target = r -> new Robot<>(
                  Controller.empty(),
                  RobotUtils.buildSensorizingFunction(targetSensorConfigName).apply(RobotUtils.buildShape("box-" + gridW + "x" + gridH))
              );
              //build evolver
              Evolver<?, UnaryOperator<Robot<?>>, DevoOutcome> evolver;
              try {
                evolver = buildEvolver(evolverName, devoFunctionName, target, fitnessFunction);
              } catch (ClassCastException | IllegalArgumentException e) {
                L.warning(String.format(
                    "Cannot instantiate %s for %s: %s",
                    evolverName,
                    devoFunctionName,
                    e
                ));
                continue;
              }
              Listener<Event<?, ? extends UnaryOperator<Robot<?>>, DevoOutcome>> listener = Listener.all(List.of(
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
                Collection<UnaryOperator<Robot<?>>> solutions = evolver.solve(
                    buildLocomotionTask(terrainName, stageMinDistance, stageMaxTime, episodeTime, random),
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
    factory.shutdown();
  }

  public static Function<UnaryOperator<Robot<?>>, DevoOutcome> buildLocomotionTask(String terrainName, double stageMinDistance, double stageMaxT, double maxT, Random random) {
    if (!terrainName.contains("-rnd")) {
      return Misc.cached(new DevoLocomotion(
          stageMinDistance, stageMaxT, maxT,
          Locomotion.createTerrain(terrainName),
          it.units.erallab.locomotion.Starter.PHYSICS_SETTINGS
      ), it.units.erallab.locomotion.Starter.CACHE_SIZE);
    }
    return r -> new DevoLocomotion(
        stageMinDistance, stageMaxT, maxT,
        Locomotion.createTerrain(terrainName.replace("-rnd", "-" + random.nextInt(10000))),
        it.units.erallab.locomotion.Starter.PHYSICS_SETTINGS
    ).apply(r);
  }

  @SuppressWarnings({"unchecked", "rawtypes"})
  private static Evolver<?, UnaryOperator<Robot<?>>, DevoOutcome> buildEvolver(String evolverName, String devoFunctionName, UnaryOperator<Robot<?>> target, Function<DevoOutcome, Double> outcomeMeasure) {
    PrototypedFunctionBuilder<?, ?> devoFunctionBuilder = null;
    for (String piece : devoFunctionName.split(it.units.erallab.locomotion.Starter.MAPPER_PIPE_CHAR)) {
      if (devoFunctionBuilder == null) {
        devoFunctionBuilder = getDevoFunctionByName(piece);
      } else {
        devoFunctionBuilder = devoFunctionBuilder.compose((PrototypedFunctionBuilder) getDevoFunctionByName(piece));
      }
    }
    return it.units.erallab.locomotion.Starter.getEvolverBuilderFromName(evolverName).build(
        (PrototypedFunctionBuilder) devoFunctionBuilder,
        target,
        PartialComparator.from(Double.class).comparing(outcomeMeasure).reversed()
    );
  }

  private static PrototypedFunctionBuilder<?, ?> getDevoFunctionByName(String name) {
    String devoHomoMLP = "devoHomoMLP-(?<ratio>\\d+(\\.\\d+)?)-(?<nLayers>\\d+)-(?<nSignals>\\d+)-(?<nInitial>\\d+)-(?<nStep>\\d+)";
    String fixedPhases = "devoPhases-(?<f>\\d+(\\.\\d+)?)-(?<nInitial>\\d+)-(?<nStep>\\d+)";
    String directNumGrid = "directNumGrid";
    Map<String, String> params;
    //devo functions
    if ((params = params(fixedPhases, name)) != null) {
      return new DevoPhasesValues(
          Double.parseDouble(params.get("f")),
          1d,
          Integer.parseInt(params.get("nInitial")),
          Integer.parseInt(params.get("nStep"))
      );
    }
    if ((params = params(devoHomoMLP, name)) != null) {
      return new DevoHomoMLP(
          Double.parseDouble(params.get("ratio")),
          Integer.parseInt(params.get("nLayers")),
          Integer.parseInt(params.get("nSignals")),
          Integer.parseInt(params.get("nInitial")),
          Integer.parseInt(params.get("nStep"))
      );
    }
    //misc
    if ((params = params(directNumGrid, name)) != null) {
      return new DirectNumbersGrid();
    }
    throw new IllegalArgumentException(String.format("Unknown devo function name: %s", name));
  }

}
