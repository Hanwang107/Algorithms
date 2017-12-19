

old_file = 'twitter_en.txt'
# old_file = 'test.txt'
new_file = 'twitter_data.txt'

with open(old_file, 'r') as input, open(new_file, 'w') as output:
    out_list = []
    new_line = ''

    for num, in_line in enumerate(input):
        if num % 2 == 0:
            new_line = ''.join(in_line.strip())
        else:
            new_line = '\t'.join([new_line, in_line])
            out_list.append(new_line)
            new_line = ''

    # Write to file
    for line in out_list:
        output.write(line)




