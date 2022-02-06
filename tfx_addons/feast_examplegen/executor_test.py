# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Tests for feast_component.executor.
"""
from google.cloud import bigquery

"""Tests for presto_component.executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
from unittest import mock

import apache_beam as beam
from apache_beam.testing import util
import datetime
import pytest

try:
  import feast
except ImportError:
  pytest.skip("feast not available, skipping", allow_module_level=True)

from tfx.v1.proto import Input

from tfx_addons.feast_examplegen import executor

import tensorflow as tf
from tfx.dsl.io import fileio
from tfx.examples.custom_components.presto_example_gen.presto_component import executor
from tfx.examples.custom_components.presto_example_gen.proto import presto_config_pb2
from tfx.proto import example_gen_pb2
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.utils import proto_utils


@beam.ptransform_fn
def _MockReadFromFeast(pipeline, query):
  del query  # Unused arg
  mock_query_results = []
  for i in range(10000):
    mock_query_result = {
        'i': None if random.randrange(10) == 0 else i,
        'f': None if random.randrange(10) == 0 else float(i),
        's': None if random.randrange(10) == 0 else str(i)
    }
    mock_query_results.append(mock_query_result)
  return pipeline | beam.Create(mock_query_results)


@beam.ptransform_fn
def _MockReadFromFeast2(pipeline, query):
  del query  # Unused arg
  mock_query_results = [{
      'timestamp': datetime.utcfromtimestamp(4.2e8),
      'i': 1,
      'i2': [2, 3],
      'b': True,
      'f': 2.0,
      'f2': [2.7, 3.8],
      's': 'abc',
      's2': ['abc', 'def']
  }]
  return pipeline | beam.Create(mock_query_results)

def _mock_get_datasource_converter(exec_properties,split_pattern):  # pylint: disable=invalid-name, unused-argument
 #todo(

class ExecutorTest(tf.test.TestCase):

  def setUp(self):
    # Mock BigQuery result schema.
    self._schema = [
        bigquery.SchemaField("timestamp",'TIMESTAMP', mode='REQUIRED'),
        bigquery.SchemaField('i', 'INTEGER', mode='REQUIRED'),
        bigquery.SchemaField('i2', 'INTEGER', mode='REPEATED'),
        bigquery.SchemaField('b', 'BOOLEAN', mode='REQUIRED'),
        bigquery.SchemaField('f', 'FLOAT', mode='REQUIRED'),
        bigquery.SchemaField('f2', 'FLOAT', mode='REPEATED'),
        bigquery.SchemaField('s', 'STRING', mode='REQUIRED'),
        bigquery.SchemaField('s2', 'STRING', mode='REPEATED'),
    ]
    super().setUp()

  # def testGetRetrievalJob(self):
  #   conn_config = presto_config_pb2.PrestoConnConfig(
  #       host='presto.localhost', max_attempts=10)
  #
  #   deseralized_conn = executor._get_retrieval_job(conn_config)
  #   truth_conn = prestodb.dbapi.connect('presto.localhost', max_attempts=10)
  #   self.assertEqual(truth_conn.host, deseralized_conn.host)
  #   self.assertEqual(truth_conn.port,
  #                    deseralized_conn.port)  # test for default port value
  #   self.assertEqual(truth_conn.auth,
  #                    deseralized_conn.auth)  # test for default auth value
  #   self.assertEqual(truth_conn.max_attempts, deseralized_conn.max_attempts)

  @mock.patch.multiple(
      executor,
      _FeastToExampleTransform=_MockReadFromFeast,
      _deserialize_conn_config=_mock_get_datasource_converter,
  )
  @mock.patch.object(bigquery, 'Client')
  def testFeastToExample(self,mock_client):
    mock_client.return_value.query.return_value.result.return_value.schema = self._schema
    with beam.Pipeline() as pipeline:
      examples = (
          pipeline | 'ToTFExample' >> executor._FeastToExampleTransform(
              exec_properties={
                  'input_config':
                      proto_utils.proto_to_json(example_gen_pb2.Input()),
                  'custom_config':
                      proto_utils.proto_to_json(example_gen_pb2.CustomConfig())
              },
              split_pattern='SELECT * FROM `fake`'))

      feature = {}
      feature['timestsamp'] =tf.train.FloatList(value=[datetime.utcfromtimestamp(4.2e8).timestamp()])
      feature['i'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[1]))
      feature['f'] = tf.train.Feature(
          float_list=tf.train.FloatList(value=[2.0]))
      feature['s'] = tf.train.Feature(
          bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes('abc')]))
      example_proto = tf.train.Example(
          features=tf.train.Features(feature=feature))
      util.assert_that(examples, util.equal_to([example_proto]))

  @mock.patch.multiple(
      executor,
      _FeastToExampleTransform=_MockReadFromFeast,
      _deserialize_conn_config=_mock_get_datasource_converter,
  )
  def testDo(self):
    output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    # Create output dict.
    examples = standard_artifacts.Examples()
    examples.uri = output_data_dir
    output_dict = {'examples': [examples]}

    # Create exe properties.
    exec_properties = {
        'input_config':
            proto_utils.proto_to_json(
                example_gen_pb2.Input(splits=[
                    example_gen_pb2.Input.Split(
                        name='bq', pattern='SELECT * FROM `fake`'),
                ])),
        'custom_config':
            proto_utils.proto_to_json(example_gen_pb2.CustomConfig()),
        'output_config':
            proto_utils.proto_to_json(
                example_gen_pb2.Output(
                    split_config=example_gen_pb2.SplitConfig(splits=[
                        example_gen_pb2.SplitConfig.Split(
                            name='train', hash_buckets=2),
                        example_gen_pb2.SplitConfig.Split(
                            name='eval', hash_buckets=1)
                    ]))),
    }

    # Run executor.
    presto_example_gen = executor.Executor()
    presto_example_gen.Do({}, output_dict, exec_properties)

    self.assertEqual(
        artifact_utils.encode_split_names(['train', 'eval']),
        examples.split_names)

    # Check Presto example gen outputs.
    train_output_file = os.path.join(examples.uri, 'Split-train',
                                     'data_tfrecord-00000-of-00001.gz')
    eval_output_file = os.path.join(examples.uri, 'Split-eval',
                                    'data_tfrecord-00000-of-00001.gz')
    self.assertTrue(fileio.exists(train_output_file))
    self.assertTrue(fileio.exists(eval_output_file))
    self.assertGreater(
        fileio.open(train_output_file).size(),
        fileio.open(eval_output_file).size())


if __name__ == '__main__':
  tf.test.main()