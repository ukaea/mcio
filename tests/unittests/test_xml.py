"""
unittest: XMLTag - attributes, nested children and indented output.
"""

import unittest
import xml.etree.ElementTree as ET

from mcio.xml import XMLTag


class MCIOUnitTestXMLTag(unittest.TestCase):
    def test_empty_tag(self):
        self.assertEqual(XMLTag('materials').as_string, '<materials />')

    def test_attributes(self):
        tag = XMLTag('material', {'id': '1', 'name': 'water'})
        root = ET.fromstring(tag.as_string)
        self.assertEqual(root.attrib, {'id': '1', 'name': 'water'})

    def test_nested_children_keep_order_and_attributes(self):
        root_tag = XMLTag('materials')
        material = XMLTag('material', {'id': '1'})
        material.add_child(XMLTag('density', {'value': '1.0', 'units': 'g/cm3'}))
        material.add_child(XMLTag('nuclide', {'name': 'H1'}))
        root_tag.add_child(material)
        root_tag.add_child(XMLTag('material', {'id': '2'}))

        root = ET.fromstring(root_tag.as_string)
        self.assertEqual([m.get('id') for m in root.findall('material')], ['1', '2'])
        first = root.find('material')
        self.assertEqual([child.tag for child in first], ['density', 'nuclide'])
        self.assertEqual(first.find('density').get('units'), 'g/cm3')

    def test_output_is_indented(self):
        root_tag = XMLTag('a')
        child = XMLTag('b')
        child.add_child(XMLTag('c'))
        root_tag.add_child(child)
        self.assertEqual(root_tag.as_string, '<a>\n  <b>\n    <c />\n  </b>\n</a>')


if __name__ == '__main__':
    unittest.main()
