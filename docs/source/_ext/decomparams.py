from docutils import nodes
from docutils.parsers.rst import Directive
import mcio
from dataclasses import fields

class DecompositionParamsTable(Directive):
    has_content = False

    def run(self):
        # 1. Instantiate the class to get actual default values
        params_inst = mcio.decomposeparams.DecompositionParameters()
        
        # 2. Build table structure (3 columns now)
        table = nodes.table()
        table['classes'].append('small-table')
        
        tgroup = nodes.tgroup(cols=3)
        table += tgroup
        tgroup += nodes.colspec(colwidth=30)
        tgroup += nodes.colspec(colwidth=15)
        tgroup += nodes.colspec(colwidth=55)

        # Header
        thead = nodes.thead()
        tgroup += thead
        header = nodes.row()
        for title in ["Parameter", "Value", "Description"]:
            entry = nodes.entry()
            entry += nodes.paragraph(text=title)
            header += entry
        thead += header

        # Body
        tbody = nodes.tbody()
        tgroup += tbody

        # 3. Logic: Iterate over fields to find definitions
        # We look for fields starting with _DEF_
        all_fields = {f.name: f for f in fields(mcio.decomposeparams.DecompositionParameters)}
        
        for name in all_fields:
            if name.startswith("_DEF_"):
                # Get the actual parameter name (e.g., AUX_BOUNDING_BOX...)
                param_name = name[5:] 
                
                # Get the value and the description string from the instance
                description = getattr(params_inst, name)
                # Handle cases where a DEF might exist without a matching value
                value = getattr(params_inst, param_name, "N/A")

                row = nodes.row()

                # Column 1: Parameter Name
                e1 = nodes.entry()
                e1 += nodes.paragraph(text=param_name)
                row += e1

                # Column 2: Value
                e2 = nodes.entry()
                e2 += nodes.paragraph(text=str(value))
                row += e2

                # Column 3: Description (The string in _DEF_...)
                e3 = nodes.entry()
                e3 += nodes.paragraph(text=str(description))
                row += e3

                tbody += row

        return [table]
    
def setup(app):
    app.add_directive("decomposition-params", DecompositionParamsTable)
    
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }