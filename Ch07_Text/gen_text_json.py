FONT_NAME = 'lato'

# Read kerning data from file
with open('./' + FONT_NAME + '.fnt', 'r') as font_file:

    # Create kerning map
    kerning_map = {};

    # Find kerning lines
    for line in font_file:
        if line.startswith("kerning "):
                        
            # Find first
            start_pos = line.find('first=')
            if start_pos != -1:
                end_pos = line.find(' ', start_pos)
                firstIndex = int(line[start_pos+6:end_pos])
                
            # Find second
            start_pos = line.find('second=')
            if start_pos != -1:
                end_pos = line.find(' ', start_pos)
                secondIndex = int(line[start_pos+7:end_pos])
                
            # Find amount
            start_pos = line.find('amount=')
            if start_pos != -1:
                end_pos = line.find(' ', start_pos)
                amount = int(line[start_pos+7:end_pos])
    
            # Update map
            if firstIndex not in kerning_map.keys():
                kerning_map[firstIndex] = {};
            kerning_map[firstIndex][secondIndex] = amount;


# Open file zto read and write font data
with open('./' + FONT_NAME + '.fnt', 'r') as font_file:

    # Open file to hold JSON data
    with open('./' + FONT_NAME + '_data.json', 'w') as out_file:
        out_file.write('[\n')

        # Iterate through characters
        for i, line in enumerate(font_file):

            if line.startswith("char "):

                # Find ID
                start_pos = line.find('id=')
                if start_pos != -1:
                    out_file.write('  {\n')
                    end_pos = line.find(' ', start_pos)
                    id = int(line[start_pos+3:end_pos])
                    out_file.write('    "id": %s,\n' % id)

                # Find x
                start_pos = line.find('x=')
                if start_pos != -1:
                    end_pos = line.find(' ', start_pos)
                    out_file.write('    "x": %s,\n' % line[start_pos+2:end_pos])

                # Find y
                start_pos = line.find('y=')
                if start_pos != -1:
                    end_pos = line.find(' ', start_pos)
                    out_file.write('    "y": %s,\n' % line[start_pos+2:end_pos])

                # Find width
                start_pos = line.find('width=')
                if start_pos != -1:
                    end_pos = line.find(' ', start_pos)
                    out_file.write('    "width": %s,\n' % line[start_pos+6:end_pos])

                # Find height
                start_pos = line.find('height=')
                if start_pos != -1:
                    end_pos = line.find(' ', start_pos)
                    out_file.write('    "height": %s,\n' % line[start_pos+7:end_pos])

                # Find xoffset
                start_pos = line.find('xoffset=')
                if start_pos != -1:
                    end_pos = line.find(' ', start_pos)
                    out_file.write('    "xoffset": %s,\n' % line[start_pos+8:end_pos])

                # Find yoffset
                start_pos = line.find('yoffset=')
                if start_pos != -1:
                    end_pos = line.find(' ', start_pos)
                    out_file.write('    "yoffset": %s,\n' % line[start_pos+8:end_pos])

                # Find xadvance
                start_pos = line.find('xadvance=')
                if start_pos != -1:
                    end_pos = line.find(' ', start_pos)
                    out_file.write('    "xadvance": %s' % line[start_pos+9:end_pos])

                # Set kerning (if applicable)
                if id in kerning_map.keys():
                    kern_str = ',\n    "kerning": { '
                    for key in kerning_map[id].keys():
                        kern_str += '"%s": %s, ' % (key, kerning_map[id][key])
                    kern_str = kern_str[:-2] + " }\n"
                    out_file.write(kern_str)
                else:
                    out_file.write('\n')

                # Check for last character
                if line.find('id=126') == -1:
                    out_file.write('  },\n')
                else:
                    out_file.write('  }\n')

        out_file.write(']\n')