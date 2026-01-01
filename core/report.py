# =============================#
#     Lint Warnings Output     #
# =============================#

def print_warnings(warnings):
        
    print("\nLint Warnings")
    print("-------------")

    if not warnings:
        print("No major issues detected")
    else:
        for warn in warnings:
            print(warn)
