def mk_find_needle(in_tens, out_tens, in_eps=0.1, out_eps=10 ** -3):
    ret = ''
    for it in range(len(in_tens)):
        ret += f"(declare-const X_{it} Real)\n"
    ret += "\n"
    for it in range(len(out_tens)):
        ret += f"(declare-const Y_{it} Real)\n"
    ret += "\n"

    ret += '; Input Box\n'
    ret += '(assert (and \n'
    for it, val in enumerate(in_tens.flatten().detach().numpy()):
        ret += f'\t (>= X_{it} {val - in_eps}) (<= X_{it} {val + in_eps})\n'
    ret += '))\n\n'

    ret += "; Output Box\n"
    ret += "(assert (and \n"
    for it, val in enumerate(out_tens.flatten().detach().numpy()):
        ret += f'\t (>= Y_{it} {val - out_eps}) (<= Y_{it} {val + out_eps})\n'
    ret += "))\n"
    return ret
