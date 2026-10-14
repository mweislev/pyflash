import numpy as np
import numbers
import operator as op
import warnings

__author__ = "Michael Weis"
__version__ = "1.0.0.1"

binop = {"+" : op.add, "-" : op.sub, "*" : op.mul, "/" : op.truediv,
    "^" : op.pow, "**" : op.pow, "|": op.or_, "&": op.and_, "//": op.floordiv,
    "<": op.lt, "<=": op.le, "==": op.eq, "!=": op.ne, ">=": op.ge, ">": op.gt}
    
# The following list is needed for checking unhashable datatypes against the
# list of operators. While operator strings itself are always hashable,
# not all stack items are (e.g. rpn input lists themselves).
binop_keys = list(binop.keys())

class rpn_program(list):
    def __init__(self, input_list, str_eval=eval):
        self.__str_eval = str_eval
        list.__init__(self, input_list)
    def eval(self):
        stack = []
        while self:
            item = self.pop(0)
            if isinstance(item, numbers.Number) or isinstance(item, np.ndarray):
                stack.append(item)
            elif item in binop_keys:
                b = stack.pop()
                a = stack.pop()
                r = binop[item](a, b)
                self.insert(0, r)
            elif isinstance(item, list):
                r = rpn_program(item).eval()
                self.insert(0, r)
            elif isinstance(item, np.ufunc):
                args = []
                for i in range(item.nin):
                    args.insert(0, stack.pop())
                r = item(*args)
                if isinstance(r, list):
                    self.insert(0, rpn_program(r).eval())
                else:
                    self.insert(0, r)
            elif callable(item):
                a = stack.pop()
                r = item(a)
                self.insert(0, r)
            elif isinstance(item, str) or isinstance(item, unicode):
                r = self.__str_eval(item)
                self.insert(0, r)
            else:
                warnings.warn('Alien RPN input: "'+str(item)+'"', RuntimeWarning)
                stack.append(item)
        self.extend(stack)
        return stack[0] if len(stack)==1 else rpn_program(stack)

# ==== TEST ====================================================================
if __name__ == '__main__':
    a1 = np.array((1,2,3))
    l1 = (a1, 2, '**')
    p1 = rpn_program(l1)
    print(p1.eval())
    p2a = rpn_program((p1, 3, '+'))
    p2b = rpn_program((rpn_program(l1), 3, '+'))
    print(p2a.eval(), p2b.eval())
    p3 = rpn_program((2, 7, np.power))
    print(p3.eval())
    f = lambda z: lambda y: lambda x: x*y*z
    p4 = rpn_program((5, 7, 11, f))
    print(p4.eval())
