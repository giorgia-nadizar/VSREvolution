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

package it.units.erallab;

import com.fasterxml.jackson.annotation.JsonAutoDetect;
import com.fasterxml.jackson.annotation.PropertyAccessor;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import it.units.erallab.hmsrobots.core.controllers.CentralizedSensing;
import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.SensingVoxel;
import it.units.erallab.hmsrobots.util.Grid;

import java.io.*;
import java.util.*;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.IntStream;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * @author eric
 * @created 2020/08/19
 * @project VSREvolution
 */
public class Utils {

  private static final Logger L = Logger.getLogger(Utils.class.getName());

  private Utils() {
  }

  public static String safelySerialize(Serializable object) {
    try (
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ObjectOutputStream oos = new ObjectOutputStream(new GZIPOutputStream(baos, true))
    ) {
      oos.writeObject(object);
      oos.flush();
      oos.close();
      return Base64.getEncoder().encodeToString(baos.toByteArray());
    } catch (IOException e) {
      L.log(Level.SEVERE, String.format("Cannot serialize due to %s", e), e);
      return "";
    }
  }

  public static <T> T safelyDeserialize(String string, Class<T> tClass) {
    try (
        ByteArrayInputStream bais = new ByteArrayInputStream(Base64.getDecoder().decode(string));
        ObjectInputStream ois = new ObjectInputStream(new GZIPInputStream(bais))
    ) {
      Object o = ois.readObject();
      return (T) o;
    } catch (IOException | ClassNotFoundException e) {
      L.log(Level.SEVERE, String.format("Cannot deserialize due to %s", e), e);
      return null;
    }
  }

  @SafeVarargs
  public static <K> List<K> concat(List<K>... lists) {
    List<K> all = new ArrayList<>();
    for (List<K> list : lists) {
      all.addAll(list);
    }
    return all;
  }

  public static <K, T> Function<K, T> ifThenElse(Predicate<K> predicate, Function<K, T> thenFunction, Function<K, T> elseFunction) {
    return k -> predicate.test(k) ? thenFunction.apply(k) : elseFunction.apply(k);
  }

  public static <K> SortedMap<Integer, K> index(List<K> list) {
    SortedMap<Integer, K> map = new TreeMap<>();
    for (int i = 0; i < list.size(); i++) {
      map.put(i, list.get(i));
    }
    return map;
  }

  public static void main(String[] args) throws JsonProcessingException {
    Random rnd = new Random();
    Grid<? extends SensingVoxel> body = it.units.erallab.hmsrobots.util.Utils.buildBody("biped-8x5-t-t");
    MultiLayerPerceptron mlp = new MultiLayerPerceptron(
        MultiLayerPerceptron.ActivationFunction.RELU,
        CentralizedSensing.nOfInputs(body),
        new int[]{(int) Math.round((double) CentralizedSensing.nOfInputs(body) * 0.65d)},
        CentralizedSensing.nOfOutputs(body)
    );
    System.out.printf("weights=%d%n", mlp.getParams().length);
    double[] ws = new double[mlp.getParams().length];
    IntStream.range(0, ws.length).forEach(i -> ws[i] = rnd.nextDouble() * 2d - 1d);
    mlp.setParams(ws);
    Robot<SensingVoxel> r = new Robot<>(
        new CentralizedSensing(body, mlp),
        body
    );
    System.out.println(safelySerialize(r).length());
    ObjectMapper om = new ObjectMapper();
    om.disable(SerializationFeature.FAIL_ON_EMPTY_BEANS);
    om.setVisibility(PropertyAccessor.FIELD, JsonAutoDetect.Visibility.ANY);
    om.setVisibility(PropertyAccessor.GETTER, JsonAutoDetect.Visibility.NONE);
    System.out.println(om.writeValueAsString(body).length());
    System.out.println(om.writeValueAsString(body));
    System.out.println(om.writeValueAsString(body.get(0, 0)));
  }

}
