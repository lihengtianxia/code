#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "chao.fang"
"""
从excel文件生成xml文件的脚本.
使用方法：
python create_xml -e xml_type1.xlsx -x params_spec.xml
"""
from __future__ import unicode_literals, print_function, division

try:
    from xml.etree import cElementTree as ET
except ImportError:
    from xml.etree import ElementTree as ET
import pandas as pd
import argparse
import os

filepath = os.path.split(os.path.realpath('__file__'))[0]


def write_input(df_input, root):
    inparams = ET.SubElement(root, 'inparams', type="map")
    for i, v in df_input.iterrows():
        if str(v["input"]) == "nan":
            break
        inparam = ET.SubElement(inparams, 'inparam')
        name = ET.SubElement(inparam, 'name')
        name.text = str(v["input"]).strip().lower()
        dName = ET.SubElement(inparam, 'dName')
        dName.text = ' '
        type = ET.SubElement(inparam, 'type')
        type.text = v["type"].strip().lower()
        defaultValue = ET.SubElement(inparam, 'defaultValue')
        defaultValue.text = "-999"
        datatype = ET.SubElement(inparam, 'datatype')
        datatype.text = v["input_datatype"].strip().lower()
        description = ET.SubElement(inparam, 'description')
        description.text = v["desc"].strip()


def write_output(df_output, root):
    outparams = ET.SubElement(root, 'outparams', type="map")
    for i, v in df_output.iterrows():
        if str(v["output"]) == "nan":
            break
        outparam = ET.SubElement(outparams, 'outparam')
        name = ET.SubElement(outparam, 'name')
        name.text = v['output']
        dName = ET.SubElement(outparam, 'dName')
        dName.text = ' '
        datatype = ET.SubElement(outparam, 'datatype')
        datatype.text = v['output_datatype']
        description = ET.SubElement(outparam, 'description')
        description.text = v["desc"].strip()


def create(excel=None, xml=None):
    if excel is None:
        excel = filepath + "/xml_type1.xlsx"
    if xml is None:
        xml = filepath + "/params_spec.xml"
    df = pd.read_excel("./xml_type1.xlsx")
    root = ET.Element('params')

    df_input = df[["input", "type", "input_datatype", "desc"]]
    write_input(df_input, root)
    df_output = df[["output", "output_datatype", "desc"]]
    write_output(df_output, root)
    tree = ET.ElementTree(root)
    tree.write(xml, encoding='utf-8',
               xml_declaration=True)
    print('---->>"{0}" is create  by "{1}"<<----'.format(xml, excel))
    print('---->>"python create_xml --help" 查看使用方法<<----')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tools for create xml file from excel file')
    parser.add_argument('-e', '--excel', nargs=1, default=[None], dest="excel",
                        help="Excel file path like: python create_xml.py -e /home/admin/xml_type1.xlsx.\n"
                             "Default is current directory xml_type1.xlsx file.")
    parser.add_argument('-x', '--xml', nargs=1, default=[None], dest="xml",
                        help="Xml file path like: python create_xml.py -x /home/admin/params_spec.xml.\n"
                             "Default is current directory params_spec.xml file")
    result = parser.parse_args()
    create(result.excel[0], result.xml[0])
