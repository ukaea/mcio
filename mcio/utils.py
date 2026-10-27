""" Utilities

General helpers: dictionary serialisation base class, dictionary printing, a
decorator factory creating output folders, notebook detection and OCP XDE
assembly parent mapping.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from OCP.TDF import TDF_Label
    from OCP.XCAFDoc import XCAFDoc_ShapeTool

import os
import functools
import inspect


try:
    import casadi
    import OCP
    from OCP.TDF import TDF_LabelSequence
except:
    pass


class SerialisableFromDict:
    """Base class allowing serialisation from and to dictionary objects.

    Subclasses must define ``_dictable_attributes`` (list of attribute names
    stored directly) and ``_dictable_sub_classes`` (mapping of attribute name to
    a :class:`SerialisableFromDict` subclass, serialised recursively).
    """
    def to_dict(self) -> dict:
        """Convert the instance to a dictionary.

        Returns
        -------
        dict
            Values of ``_dictable_attributes`` and the ``to_dict`` output of each
            ``_dictable_sub_classes`` attribute, keyed by attribute name.
        """
        data: dict = {k: getattr(self, k) for k in self._dictable_attributes}
        
        for sub_class_attr in self._dictable_sub_classes:
            data[sub_class_attr] = getattr(self, sub_class_attr).to_dict()

        return data

    @classmethod
    def from_dict(cls, data: dict) -> SerialisableFromDict:
        """Create an instance from a dictionary.

        Keys not listed in ``_dictable_attributes`` / ``_dictable_sub_classes``
        and values of None are ignored. ``__post_init__`` is called on the new
        instance if defined.

        Parameters
        ----------
        data : dict
            Dictionary of attribute values, as produced by :meth:`to_dict`.

        Returns
        -------
        SerialisableFromDict
            New instance of ``cls`` populated from ``data``.

        Raises
        ------
        TypeError
            If ``data`` is not a dictionary.
        """
        # create blank instance of this class
        class_instance = cls()

        # check input is dictionary type
        if not isinstance(data, dict):
            msg = f'from_dictionary type should be dict, not {type(data)}'
            raise TypeError(msg)
        
        # filter for values which are not none
        data_filtered_attr = {k: v for k, v in data.items() if k in class_instance._dictable_attributes and not isinstance(v, type(None))}
        data_filtered_subcls = {k: v for k, v in data.items() if k in class_instance._dictable_sub_classes and not isinstance(v, type(None))}

        # set the values within the new class instance
        for k, v in data_filtered_attr.items():
            setattr(class_instance, k, v)

        for k, v in data_filtered_subcls.items():
            sub_class_instance = class_instance._dictable_sub_classes[k].from_dict(v)
            setattr(class_instance, k, sub_class_instance)

        # run post init for the new class instance
        if hasattr(class_instance, '__post_init__'):
            class_instance.__post_init__()
        
        # return the class instance
        return class_instance

def dict_as_string(dict: dict, level: int = 1) -> str:
    """Return a (possibly nested) dictionary as an indented string.

    Parameters
    ----------
    dict : dict
        Dictionary to convert.
    level : int, optional
        Indentation level (number of ``-`` characters), by default 1

    Returns
    -------
    str
        String representation of the dictionary.
    """
    flag_ocp = False
    
    output_string = ''

    for k, v in dict.items():

        indent = '-' * level

        string_key = f'{indent} k={k}\n'
        output_string += string_key

        if isinstance(v, type(dict)):
            output_string = dict_as_string(v, level=level+1)
        else:
            string_val = f'{indent + indent[0]} {v}'
            output_string += string_val

    return output_string

def check_folder_exists_for_file_path(
    param_name: str,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Create a decorator which ensures the folder of a file path argument exists.

    Decorator factory: the produced decorator looks up the argument
    ``param_name`` of the decorated function and, if its directory does not
    exist, creates it before calling the function.

    Parameters
    ----------
    param_name : str
        Name of the decorated function's parameter holding the file path.

    Returns
    -------
    Callable[[Callable[..., Any]], Callable[..., Any]]
        Decorator applying the folder check.
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap a function with the folder existence check.

        Parameters
        ----------
        func : Callable[..., Any]
            Function taking a file path parameter named ``param_name``.

        Returns
        -------
        Callable[..., Any]
            Wrapped function.
        """
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Create the file path's folder if required, then call the function.

            Parameters
            ----------
            *args : Any
                Positional arguments passed to the wrapped function.
            **kwargs : Any
                Keyword arguments passed to the wrapped function.

            Returns
            -------
            Any
                Result of the wrapped function.
            """
            # Get function signature so we can map args → parameter names
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            # Extract the file path from the specified parameter
            file_path = bound.arguments[param_name]

            # Split into directory + filename
            directory, _ = os.path.split(file_path)

            # Create directory if needed
            if directory and not os.path.isdir(directory):
                os.makedirs(directory, exist_ok=True)

            return func(*args, **kwargs)
        return wrapper
    return decorator

def in_notebook() -> bool:
    """Check whether the code is running in a Jupyter notebook or lab.

    Returns
    -------
    bool
        True if the IPython shell is a ``ZMQInteractiveShell``, else False.
    """
    try:
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__
        return shell == "ZMQInteractiveShell"  # Jupyter notebook or lab
    except Exception:
        return False

def build_parent_map(shape_tool: XCAFDoc_ShapeTool) -> dict[TDF_Label, TDF_Label]:
    """Build a map from each assembly component label to its parent label.

    Parameters
    ----------
    shape_tool : XCAFDoc_ShapeTool
        XDE shape tool of the document, walked from its free (top-level) shapes.

    Returns
    -------
    dict[TDF_Label, TDF_Label]
        Mapping of child label to parent label.
    """
    parent_map: dict[TDF_Label, TDF_Label] = {}  # child_label -> parent_label

    def walk(label: TDF_Label) -> None:
        """Recursively record the components of a label in ``parent_map``.

        Parameters
        ----------
        label : TDF_Label
            Label whose components are visited.
        """
        children = TDF_LabelSequence()
        shape_tool.GetComponents_s(label, children)

        for i in range(children.Length()):
            child = children.Value(i+1)
            parent_map[child] = label
            walk(child)

    # Start from free shapes (top-level assemblies)
    roots = TDF_LabelSequence()
    shape_tool.GetFreeShapes(roots)

    for i in range(roots.Length()):
        walk(roots.Value(i+1))

    return parent_map
