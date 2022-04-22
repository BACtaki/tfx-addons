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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from google.cloud import bigquery
from google.protobuf.struct_pb2 import Struct

"""Tests for presto_component.executor."""

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
from tfx.proto import example_gen_pb2
from tfx.types import artifact_utils
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

def _mock_load_custom_config(custom_config):
    repo_config = feast.RepoConfig(provider='local', project='default')
    repo_conf = repo_config.json(exclude={"repo_path"}, exclude_unset=True)
    feature_refs=['feature1', 'feature2']

    return {executor._REPO_CONFIG_KEY:repo_conf, executor._FEATURE_KEY: feature_refs}
def _mock_get_datasource_converter(exec_properties,split_pattern):  # pylint: disable=invalid-name, unused-argument
    ...
def _mock_get_retrieval_job(en, split_pattern):  # pylint: disable=invalid-name, unused-argument
    ...


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

  def testLoadCustomConfig(self):
    repo_config = feast.RepoConfig(provider='local', project='default')
    repo_conf = repo_config.json(exclude={"repo_path"}, exclude_unset=True)
    feature_refs=['feature1', 'feature2']
    config_struct = Struct()
    config_struct.update({executor._REPO_CONFIG_KEY: repo_conf, executor._FEATURE_KEY: feature_refs})
    custom_config_pbs2 = example_gen_pb2.CustomConfig()
    custom_config_pbs2.custom_config.Pack(config_struct)
    custom_config  = proto_utils.proto_to_json(custom_config_pbs2)

    deseralized_conn = executor._load_custom_config(custom_config)
    truth_config = _mock_load_custom_config("dummy")
    self.assertEqual(deseralized_conn, truth_config)

  @mock.patch.multiple(
      executor,
      _load_custom_config=_mock_load_custom_config,
  )

  def testGetRetrievalJob(self):
    ...

  @mock.patch.multiple(
      executor,
      _load_custom_config=_mock_load_custom_config,
      _get_retrieval_job=_mock_get_retrieval_job
  )

  def testGetDatasourceConverter(self):
    ...

  @mock.patch.multiple(
      executor,
      _load_custom_config=_mock_load_custom_config,
      _get_datasource_converter=_mock_get_datasource_converter
  )
  @mock.patch.object(bigquery, 'Client')
  def testFeastToExample(self,mock_datasource):
    ...

  @mock.patch.multiple(
      executor,
      _load_custom_config=_mock_load_custom_config,
      _get_datasource_converter=_mock_get_datasource_converter
  )
  def testDo(self):
    ...


if __name__ == '__main__':
  tf.test.main()