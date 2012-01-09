import re
import sys

def swap(tupl):
    return (tupl[1], tupl[0])

class Token:
    def __init__(self, typ, lexeme, line, col):
        assert(isinstance(typ, str))
        assert(isinstance(lexeme, str))
        assert(isinstance(line, int))
        assert(isinstance(col, int))
        
        self.typ = typ
        self.lexeme = lexeme
        self.line = line
        self.col = col

    def __str__(self):
        return "%s(%s):%d:%d" % (self.typ, self.lexeme, self.line, self.col)

class Lexer:
    def __init__(self, source):
        assert(isinstance(source, str))
        self.source = source
        self.line = 1
        self.col = 0

        self.stream_pos = 0

    def peek(self):
        while self.stream_pos < len(self.source) and \
                self.source[self.stream_pos] in [" ", "\t", "\r", "\n"]:
            if self.source[self.stream_pos] == "\n":
                self.col = 0
                self.line += 1
                self.stream_pos += 1
            else:
                self.col += 1
                self.stream_pos += 1

        if self.stream_pos == len(self.source):
            return Token("eof", "", self.line, self.col)

        rex = re.compile(r"^([;,(){}+*/-])|(:=|!=|<=|>=|<|>|=)|([0-9]+)|([a-zA-Z][a-zA-Z0-9_]*)")
        m = rex.search(self.source[self.stream_pos:])
        if not m:
            raise SyntaxError("Unexpected character: %s" % self.source[self.source_offset])
        if m.group()[0] in "0 1 2 3 4 5 6 7 8 9".split():
            return Token("number", m.group(), self.line, self.col)
        elif m.group() in "; , ( ) { } := + - / * != <= >= < > =".split():
            return Token(m.group(), m.group(), self.line, self.col)
        else:
            s = m.group()
            if s in "int bool".split():
                return Token("type", s, self.line, self.col)
            elif s == "return":
                return Token("return", s, self.line, self.col)
            elif s in "true false".split():
                return Token("boolconst", s, self.line, self.col)
            elif s in "and or not".split():
                return Token(s, s, self.line, self.col)
            elif s in "if else then fi while".split():
                return Token(s, s, self.line, self.col)
            else:
                return Token("id", s, self.line, self.col)

    def next(self):
        t = self.peek()
        if t.typ != "eof":
            self.col += len(t.lexeme)
            self.stream_pos += len(t.lexeme)
        if t.typ == "id" and len(t.lexeme) > 8:
            print("%d:%d warning: identifier %s exceeds length of 8 characters and will be truncated" % (t.line, t.col, t.lexeme), file=sys.stderr)
        t.lexeme = t.lexeme[:8]
            
        return t

class SymbolEntry:
    def __init__(self, ret_type, id_tok, prototype=None):
        self.ret_type = ret_type
        self.id_tok = id_tok
        self.prototype = prototype

    def is_var(self):
        return self.prototype == None

    def is_func(self):
        return self.prototype != None

    def __str__(self):
        return "%s(%s):%d:%d" % (self.ret_type, self.id_tok.lexeme, self.id_tok.line, self.id_tok.col)

class SymbolTable:
    def __init__(self):
        self.frames = []
        self.push_frame()

    def push_frame(self):
        self.frames.insert(0, [])

    def pop_frame(self):
        self.frames = self.frames[1:]

    def is_identifier_local(self, identifier):
        for e in self.frames[0]:
            if e.id_tok.lexeme == identifier:
                return True
        return False

    def find_entry_by_id(self, identifier):
        for frame in self.frames:
            for e in frame:
                if e.id_tok.lexeme == identifier:
                    return e
        return None

    def register(self, se):
        assert(isinstance(se, SymbolEntry))
        if self.is_identifier_local(se.id_tok.lexeme):
            old_se = self.find_entry_by_id(se.id_tok.lexeme)
            raise SyntaxError("%d:%d error: redefining symbol %s. First defined here: %d:%d" %
                              (se.id_tok.line, se.id_tok.col, se.id_tok.lexeme, old_se.id_tok.line, old_se.id_tok.col))
        else:
            self.frames[0].append(se)

    def __str__(self):
        return "[%s]" % ("; ".join([ "[%s]" % (", ".join([str(e) for e in frame]))  for frame in self.frames]))

class ReturnAST:
    def __init__(self, expr, ret_tok):
        self.expr = expr
        self.tok = ret_tok

    def __str__(self):
        return "(return %s)" % str(self.expr)

    def get_ret_type(self):
        return self.expr.get_ret_type()

class VarDeclAST:
    def __init__(self, typ, var_name):
        self.typ = typ
        self.var_name = var_name

    def __str__(self):
        return "(defvar %s %s)" % (self.var_name, self.typ)

    def get_ret_type(self):
        return None

class BlockAST:
    def __init__(self):
        self.exprs = []

    def __str__(self):
        return "(progn %s)" % (" ".join([str(exp) for exp in self.exprs]))

    def get_ret_type(self):
        return None

class AssignAST:
    def __init__(self, var_name, assign_expr):
        self.var_name = var_name
        self.assign_expr = assign_expr

    def __str__(self):
        return "(setf %s %s)" % (self.var_name, str(self.assign_expr))

    def get_ret_type(self):
        return None

class FuncCallAST:
    def __init__(self, func_name, arg_asts, symbol_table):
        self.func_name = func_name
        self.arg_asts = arg_asts
        self.ret_type = symbol_table.find_entry_by_id(func_name).ret_type

    def __str__(self):
        return "(funcall %s %s)" % (self.func_name, " ".join([str(ast) for ast in self.arg_asts]))

    def get_ret_type(self):
        return self.ret_type
            
class VarRefAST:
    def __init__(self, var_name, symbol_table):
        self.var_name = var_name
        self.ret_type = symbol_table.find_entry_by_id(self.var_name).ret_type

    def __str__(self):
        return self.var_name

    def get_ret_type(self):
        return self.ret_type

class BoolConstAST:
    def __init__(self, val):
        self.value = val

    def __str__(self):
        if self.value == "false":
            return "NIL"
        else:
            return "T"

    def get_ret_type(self):
        return "bool"

class NumberAST:
    def __init__(self, intgr):
        self.value = intgr

    def __str__(self):
        return str(self.value)

    def get_ret_type(self):
        return "int"

class SignedAST:
    def __init__(self, sign, expr):
        self.sign = sign
        self.expr = expr

    def __str__(self):
        if self.sign == "-":
            return "(negate %s)" % str(self.expr)
        else:
            return str(self.expr)

    def get_ret_type(self):
        return "int"

class NotAST:
    def __init__(self, expr):
        self.expr = expr

    def __str__(self):
        return "(not %s)" % str(self.expr)

class MathOpAST:
    def __init__(self, op, lhs_expr, rhs_expr):
        self.op = op
        self.lhs_expr = lhs_expr
        self.rhs_expr = rhs_expr
        
    def __str__(self):
        return "(%s %s %s)" % (self.op, str(self.lhs_expr), str(self.rhs_expr))

    def get_ret_type(self):
        return "int"

class BoolOpAST:
    def __init__(self, op, lhs_expr, rhs_expr):
        self.op = op
        self.lhs_expr = lhs_expr
        self.rhs_expr = rhs_expr
        
    def __str__(self):
        return "(%s %s %s)" % (self.op, str(self.lhs_expr), str(self.rhs_expr))

    def get_ret_type(self):
        return "bool"

class AndNotAST:
    def __init__(self, op, rhs_expr):
        self.op = op
        self.rhs_expr = rhs_expr

    def __str__(self):
        return "(%s %s)" % (self.op, str(self.rhs_expr))

    def get_ret_type(self):
        return "bool"

class RelOpAST:
    def __init__(self, op, lhs_expr, rhs_expr):
        self.op = op
        self.lhs_expr = lhs_expr
        self.rhs_expr = rhs_expr

    def __str__(self):
        return "(%s %s %s)" % (self.op, str(self.lhs_expr), str(self.rhs_expr))

    def get_ret_type(self):
        return "bool"

class IfAST:
    def __init__(self, chk_ast, then_ast, else_ast=None):
        self.chk_ast = chk_ast
        self.then_ast = then_ast
        self.else_ast = else_ast

    def __str__(self):
        if self.else_ast:
            return "(if %s %s %s)" % (str(self.chk_ast), str(self.then_ast), str(self.else_ast))
        else:
            return "(when %s %s)" % (str(self.chk_ast), str(self.then_ast))

    def get_ret_type(self):
        return None

class WhileAST:
    def __init__(self, chk_ast, loop_ast):
        self.chk_ast = chk_ast
        self.loop_ast = loop_ast

    def __str__(self):
        return "(while %s %s)" % (str(self.chk_ast), str(self.loop_ast))

class FuncDeclAST:
    def __init__(self, ret_typ, func_name, params, block):
        self.ret_typ = ret_typ
        self.func_name = func_name
        self.params = params
        self.block = block

    def __str__(self):
        return "(defun %s %s (%s) %s)" % (self.func_name, self.ret_typ, " ".join(["(%s %s)" % swap(p)  for p in self.params]), str(self.block))

    def get_ret_type(self):
        return None

    def check_return_types(self, block= None):
        if not block:
            block = self.block

        for expr in block.exprs:
            if isinstance(expr, ReturnAST):
                if expr.get_ret_type() != self.ret_typ:
                    raise SyntaxError((expr.get_ret_type(), expr.tok))
            elif isinstance(expr, BlockAST):
                self.check_return_types(expr)
            elif isinstance(expr, IfAST):
                if isinstance(expr.then_ast, BlockAST):
                    self.check_return_types(expr.then_ast)
                elif isinstance(expr.then_ast, ReturnAST):
                    if expr.then_ast.get_ret_type() != self.ret_typ:
                        raise SyntaxError((expr.then_ast.get_ret_type(), expr.then_ast.tok))
                if expr.else_ast and isinstance(expr.else_ast, BlockAST):
                    self.check_return_types(expr.else_ast)
                elif expr.else_ast and isinstance(expr.else_ast, ReturnAST):
                    if expr.else_ast.get_ret_type() != self.ret_typ:
                        raise SyntaxError((expr.else_ast.get_ret_type(), expr.else_ast.tok))
            elif isinstance(expr, WhileAST):
                if isinstance(expr.loop_ast, BlockAST):
                    self.check_return_types(expr.loop_ast)
                elif isinstance(expr.loop_ast, ReturnAST):
                    if expr.loop_ast.get_ret_type() != self.ret_typ:
                        raise SyntaxError((expr.loop_ast.get_ret_type(), expr.loop_ast.tok))

    def check_return_paths(self, block = None):
        if not block:
            block = self.block
        
        for expr in block.exprs:
            if isinstance(expr, ReturnAST):
                return True
            elif isinstance(expr, BlockAST):
                if self.check_return_paths(expr):
                    return True
            elif isinstance(expr, IfAST):
                if expr.else_ast != None:
                    then_chk = False
                    if isinstance(expr.then_ast, ReturnAST):
                        then_chk = True
                    elif isinstance(expr.then_ast, BlockAST):
                        then_chk = self.check_return_paths(expr.then_ast)

                    if then_chk:
                        if isinstance(expr.else_ast, ReturnAST):
                            return True
                        elif isinstance(expr.else_ast, BlockAST) and self.check_return_paths(expr.else_ast):
                            return True
        return False

class Parser:
    def __init__(self, lexer):
        assert(isinstance(lexer, Lexer))
        self.lexer = lexer
        self.symbol_table = SymbolTable()
        self.root_block = BlockAST()

    def assert_tok_types(self, tok, *types):
        assert(len(types) > 0)
        if tok.typ not in types:
            expect_msg = "%s or %s" % (", ".join(types[1:]),
                                       types[0])
            raise SyntaxError("%d:%d error: Unexpected token '%s'. Expected %s." %
                              (tok.line, tok.col, tok.lexeme, expect_msg))

    def parse(self):
        return self.parse_program()

    def fetch_token(self, *types):
        tok = self.lexer.next()
        if types:
            self.assert_tok_types(tok, *types)
        return tok

    def peek_token(self):
        return self.lexer.peek()

    def parse_program(self):
        type_tok = self.fetch_token("type")
        id_tok = self.fetch_token("id")
        dec_tok = self.fetch_token(",",";","(")
        if dec_tok.typ == ";":
            self.symbol_table.register(SymbolEntry(type_tok.lexeme, id_tok))
            self.root_block.exprs.append(VarDeclAST(type_tok.lexeme, id_tok.lexeme))
            self.parse_program()
        elif dec_tok.typ == ",":
            id_toks = [id_tok]
            while True:
                id_tok = self.fetch_token("id")
                id_toks.append(id_tok)
                dec_tok = self.fetch_token(",", ";")
                if dec_tok.typ == ";":
                    break

            for id_tok in id_toks:
                self.symbol_table.register(SymbolEntry(type_tok.lexeme, id_tok))
                self.root_block.exprs.append(VarDeclAST(type_tok.lexeme, id_tok.lexeme))
            self.parse_program()
        elif dec_tok.typ == "(":
            self.params = []
            self.symbol_table.register(SymbolEntry(type_tok.lexeme, id_tok, self.params))
            func_ast = FuncDeclAST(type_tok.lexeme, id_tok.lexeme, [], None)
            func_tok = id_tok
            self.symbol_table.push_frame()
            if self.peek_token().typ == ")":
                func_ast.block = self.parse_block()
            else:
                while True:               
                    typ_tok = self.fetch_token("type")
                    id_tok = self.fetch_token("id")
                    self.params.append((typ_tok, id_tok))
                    if self.peek_token().typ != ",":
                        break
                    self.fetch_token(",")
                self.fetch_token(")")
                for ttok, itok in self.params:
                    self.symbol_table.register(SymbolEntry(ttok.lexeme, itok))
                    func_ast.params.append((ttok.lexeme, itok.lexeme))
                func_ast.block = self.parse_block()
                self.root_block.exprs.append(func_ast)
            try:
                func_ast.check_return_types()
            except SyntaxError as e:
                raise SyntaxError("%d:%d error: function %s expects %s type, but returning %s" % (e.args[0][1].line, e.args[0][1].col, func_tok.lexeme, type_tok.lexeme, e.args[0][0]))
            if not func_ast.check_return_paths():
                raise SyntaxError("%d:%d error: not all paths in function %s return a value" % (func_tok.line, func_tok.col, func_tok.lexeme))
            self.symbol_table.pop_frame()
            self.parse_program_func_decls()
            
    def parse_program_func_decls(self):
        while self.peek_token().typ != "eof":
            decl = self.parse_func_decl()
            self.root_block.exprs.append(decl)

    def parse_func_decl(self):
        type_tok = self.fetch_token("type")
        id_tok = self.fetch_token("id")
        self.fetch_token("(")

        self.params = []
        self.symbol_table.register(SymbolEntry(type_tok.lexeme, id_tok, self.params))
        func_ast = FuncDeclAST(type_tok.lexeme, id_tok.lexeme, [], None)
        func_tok = id_tok
        self.symbol_table.push_frame()
        if self.peek_token().typ == ")":
            self.fetch_token(")")
            func_ast.block = self.parse_block()
        else:
            while True:               
                typ_tok = self.fetch_token("type")
                id_tok = self.fetch_token("id")
                self.params.append((typ_tok, id_tok))
                if self.peek_token().typ != ",":
                    break
                self.fetch_token(",")
            self.fetch_token(")")
            for ttok, itok in self.params:
                self.symbol_table.register(SymbolEntry(ttok.lexeme, itok))
                func_ast.params.append((ttok.lexeme, itok.lexeme))
            func_ast.block = self.parse_block()
            self.root_block.exprs.append(func_ast)
        try:
            func_ast.check_return_types()
        except SyntaxError as e:
            raise SyntaxError("%d:%d error: function %s expects %s type, but returning %s" % (e.args[0][1].line, e.args[0][1].col, func_tok.lexeme, type_tok.lexeme, e.args[0][0]))
        if not func_ast.check_return_paths():
            raise SyntaxError("%d:%d error: not all paths in function %s return a value" % (func_tok.line, func_tok.col, func_tok.lexeme))
        self.symbol_table.pop_frame()
        return func_ast
            
    def parse_block(self):
        ast = BlockAST()
        self.fetch_token("{")
        while self.peek_token().typ == "type":
            typ_tok = self.fetch_token("type")
            id_tok = self.fetch_token("id")
            id_toks = [id_tok]
            while self.peek_token().typ != ";":
                self.fetch_token(",")
                id_toks.append(self.fetch_token("id"))
            self.fetch_token(";")
            for tok in id_toks:
                self.symbol_table.register(SymbolEntry(typ_tok.lexeme, tok))
                ast.exprs.append(VarDeclAST(typ_tok.lexeme, tok.lexeme))
        while self.peek_token().typ != "}":
            ast.exprs.append(self.parse_stmt())
        self.fetch_token("}")
        return ast

    def parse_stmt(self):
        peek_tok = self.peek_token()
        if peek_tok.typ == "id" or peek_tok.typ == "return":
            return self.parse_simple_stmt()
        else:
            return self.parse_struct_stmt()

    def parse_simple_stmt(self):
        tok = self.fetch_token("return", "id")
        if tok.typ == "return":
            result = ReturnAST(self.parse_expr(), tok)
            self.fetch_token(";")
            return result
        else:
            peek = self.peek_token()
            if peek.typ == "(":
                result = self.parse_fun_call(tok)
                self.fetch_token(";")
                return result
            elif peek.typ == ":=": # :=
                result = self.parse_assignment(tok)
                self.fetch_token(";")
                return result

    def parse_fun_call(self, fun_tok):
        sym = self.symbol_table.find_entry_by_id(fun_tok.lexeme)
        if not sym:
            raise SyntaxError("%d:%d error: unknown symbol %s" % (fun_tok.line, fun_tok.col, fun_tok.lexeme))
        if sym.prototype == None:
            raise SyntaxError("%d:%d error: symbol %s is not a function" % (fun_tok.line, fun_tok.col, fun_tok.lexeme))
        self.fetch_token("(")
        arg_asts = []
        if self.peek_token().typ != ")":
            while True:
                arg_asts.append(self.parse_expr())
                dec_token = self.fetch_token(")", ",")
                if dec_token.typ == ")":
                    break
        self.fetch_token(";")
        
        if len(sym.prototype) != len(arg_asts):
            raise SyntaxError("%d:%d error: function %s requires %d arguments, but %d are being passed." % (fun_tok.line, fun_tok.col, fun_tok.lexeme, len(sym.prototype), len(arg_asts)))
        
        for typ, ast, i in zip(sym.prototype, arg_asts, range(len(arg_asts))):
            if typ[0].lexeme != ast.get_ret_type():
                raise SyntaxError("%d:%d error: argument %d requires type %s" % (fun_tok.line, fun_tok.col, i, typ[0].lexeme))

        return FuncCallAST(fun_tok.lexeme, arg_asts, self.symbol_table)

    def parse_expr(self):
        lhs = self.parse_simple_expr()
        dec_tok = self.peek_token()
        if dec_tok.typ in "= != < <= > >=".split():
            self.fetch_token()
            rhs = self.parse_simple_expr()
            if dec_tok.typ in "< <= > >=":
                if lhs.get_ret_type() != "int":
                    raise SyntaxError("%d:%d error: LHS value must be int typed" % (dec_tok.line, dec_tok.col))
                if rhs.get_ret_type() != "int":
                    raise SyntaxError("%d:%d error: RHS value must be int typed" % (dec_tok.line, dec_tok.col))
            return RelOpAST(dec_tok.lexeme, lhs, rhs)
        else:
            return lhs

    def parse_simple_expr(self):
        lhs = self.parse_term()
        dec_tok = self.peek_token()
        if dec_tok.typ in "+ - or".split():
            self.fetch_token("+", "-", "or")
            rhs = self.parse_term()
            if dec_tok.typ in "+ -".split():
                if lhs.get_ret_type() != "int":
                    raise SyntaxError("%d:%d error: LHS must be int typed" % (dec_tok.line, dec_tok.col))
                if rhs.get_ret_type() != "int":
                    raise SyntaxError("%d:%d error: RHS must be int typed" % (dec_tok.line, dec_tok.col))
                return MathOpAST(dec_tok.lexeme, lhs, rhs)
            else:
                if lhs.get_ret_type() != "bool":
                    raise SyntaxError("%d:%d error: LHS must be bool typed" % (dec_tok.line, dec_tok.col))
                if rhs.get_ret_type() != "bool":
                    raise SyntaxError("%d:%d error: RHS must be bool typed" % (dec_tok.line, dec_tok.col))
                return BoolOpAST(dec_tok.lexeme, lhs, rhs)
        else:
            return lhs

    def parse_term(self):
        lhs = self.parse_factor()
        dec_tok = self.peek_token()
        if dec_tok.typ in "* / and".split():
            self.fetch_token("*", "/", "and")
            rhs = self.parse_factor()
            if dec_tok.typ in "* /".split():
                if lhs.get_ret_type() != "int":
                    raise SyntaxError("%d:%d error: LHS must be int typed" % (dec_tok.line, dec_tok.col))
                if rhs.get_ret_type() != "int":
                    raise SyntaxError("%d:%d error: RHS must be int typed" % (dec_tok.line, dec_tok.col))
                return MathOpAST(dec_tok.lexeme, lhs, rhs)
            else:
                if lhs.get_ret_type() != "bool":
                    raise SyntaxError("%d:%d error: LHS must be bool typed" % (dec_tok.line, dec_tok.col))
                if rhs.get_ret_type() != "bool":
                    raise SyntaxError("%d:%d error: RHS must be bool typed" % (dec_tok.line, dec_tok.col))
                return BoolOpAST(dec_tok.lexeme, lhs, rhs)
        else:
            return lhs
        
    def parse_factor(self):
        tok = self.fetch_token("number", "boolconst",
                               "id", "(",
                               "+", "-", "not")
        if tok.typ == "boolconst":
            return BoolConstAST(tok.lexeme)
        elif tok.typ == "(":
            ast = self.parse_expr()
            self.fetch_token(")")
            return ast
        elif tok.typ in "+ -".split():
            rhs = self.parse_factor()
            if rhs.get_ret_type() != "int":
                raise SyntaxError("%d:%d error: attempting to apply sign to a %s typed expression" % (tok.line, tok.col,
                                                                                                      rhs.get_ret_type()))
            return SignedAST(tok.lexeme, rhs)
        elif tok.typ == "not":
            rhs = self.parse_factor()
            if rhs.get_ret_type() != "bool":
                raise SyntaxError("%d:%d error: attempting to negate a %s typed expression" % (tok.line, tok.col,
                                                                                               rhs.get_ret_type()))
            return AndNotAST("not", rhs)
        elif tok.typ == "number":
            rhs = NumberAST(int(tok.lexeme))
            return rhs
        else:
            dec_tok = self.peek_token()
            if dec_tok.typ == "(":
                return self.parse_fun_call(tok)
            else:
                return VarRefAST(tok.lexeme, self.symbol_table)

    def parse_assignment(self, id_tok):
        lhs = VarRefAST(id_tok.lexeme, self.symbol_table)
        self.fetch_token(":=")
        rhs = self.parse_expr()
        if lhs.get_ret_type() != rhs.get_ret_type():
            raise SyntaxError("%d:%d error: attempting to assign %s typed expression to %s typed variable %s" % (id_tok.line, id_tok.col,
                                                                                                                 rhs.get_ret_type(), lhs.get_ret_type(),
                                                                                                                 id_tok.lexeme))
        
        return AssignAST(id_tok.lexeme, rhs)

    def parse_struct_stmt(self):
        dec_tok = self.peek_token()
        if dec_tok.typ == "{":
            self.symbol_table.push_frame()
            result = self.parse_block()
            self.symbol_table.pop_frame()
            return result
        elif dec_tok.typ == "if":
            return self.parse_cond()
        elif dec_tok.typ == "while":
            return self.parse_loop()
        else:
            self.fetch_token("{ if while".split())

    def parse_loop(self):
        tok = self.fetch_token("while")
        self.fetch_token("(")
        chk_ast = self.parse_expr()
        if chk_ast.get_ret_type() != "bool":
            raise SyntaxError("%d:%d error: while clause must be bool typed" % (tok.line, tok.col))
        self.fetch_token(")")
        loop_ast = self.parse_stmt()
        return WhileAST(chk_ast, loop_ast)

    def parse_cond(self):
        self.fetch_token("if")
        tok = self.fetch_token("(")
        chk_ast = self.parse_expr()
        if chk_ast.get_ret_type() != "bool":
            raise SyntaxError("%d:%d error: if clause must be bool typed" % (tok.line, tok.col))
        tok = self.fetch_token(")")
        tok = self.fetch_token("then")
        then_ast = self.parse_stmt()
        dec_tok = self.fetch_token("fi", "else")
        if dec_tok.typ == "else":
            else_ast = self.parse_stmt()
            self.fetch_token("fi")
            return IfAST(chk_ast, then_ast, else_ast)
        else:
            return IfAST(chk_ast, then_ast)

p = Parser(Lexer("""int foo, bar, moo;
                    bool boo;
                    int func(int arg1) {
                        int local1; bool b2, b3;

                        return 5;
                    }

                    int func2() {
                        int bar; bool b;
                        b := bar < bar;
                        { int bar; bar := 0 + (2+(2*2)*5);}

                        return 6;

                        if (true) then
                            return bar + 3;
                        else
                            { while (bar > 0) {
                                bar := bar - 1;
                                b := not b;
                              }
                              if (true) then
                                  return 5;
                              else
                                  return 6;
                        fi
                              return 2 + (2*2);
                            }
                        fi
                    }

                    bool func34567890123() {return true;}

                    bool bfunc(bool b)
                    {
                        return not boo and b;
                    }"""))
p.parse()
print(p.root_block)
