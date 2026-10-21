"""Minimal XML tag tree used to build XML output (e.g. GDML) as strings."""

from __future__ import annotations

import xml
import xml.etree.ElementTree as ET


class XMLTag:
    """A node in an XML tree with a name, attributes and child tags.

    Parameters
    ----------
    name : str
        Tag name.
    attributes : dict[str, str], optional
        Mapping of attribute names to string values, by default None
    """

    def __init__(self, name: str, attributes: dict[str, str] | None = None) -> None:
        """Initialise the XMLTag."""
        self._name: str = name
        self._attributes: dict[str, str] | None = attributes 
        self._children: list[XMLTag] = []
        
    def add_child(self, child: XMLTag) -> None:
        """Append a child tag.

        Parameters
        ----------
        child : XMLTag
            Tag to add as a child of this tag.
        """
        self._children.append(child)
        
    @property
    def as_string(self) -> str:
        """Return this tag and all descendants as an indented XML string.

        Returns
        -------
        str
            UTF-8 decoded XML text with two-space indentation.
        """
        # set up root tag
        root_tag = ET.Element(self._name)
        
        if self._attributes:
            for k, v in self._attributes.items():
                root_tag.set(k, v)
        
        # iterate through children
        for child in self._children:
            child.assign_parent(root_tag)
        
        # get string from XML tags
        ET.indent(root_tag, space="  ", level=0)

        return ET.tostring(root_tag).decode('utf-8')
    
    def assign_parent(self, parent: ET.Element) -> None:
        """Create this tag (and its children, recursively) as a sub-element of parent.

        Parameters
        ----------
        parent : xml.etree.ElementTree.Element
            Element under which this tag is created.
        """
        # set up xml tag with attributes
        sub_element = ET.SubElement(parent, self._name)
        if self._attributes:
            for k, v in self._attributes.items():
                sub_element.set(k, v)
                
        for child in self._children:
            child.assign_parent(sub_element)
