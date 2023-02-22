from triton import *
from dumpulator import Dumpulator
from unicorn import *
from unicorn.x86_const import *
import os

def get_map_prot(name):
    if name == "UNDEFINED":
        return UC_PROT_NONE
    elif name == "PAGE_EXECUTE":
        return UC_PROT_EXEC
    elif name == "PAGE_EXECUTE_READ":
        return UC_PROT_READ | UC_PROT_EXEC
    elif name == "PAGE_EXECUTE_READWRITE":
        return UC_PROT_ALL
    elif name == "PAGE_READWRITE":
        return UC_PROT_READ | UC_PROT_WRITE
    elif name == "PAGE_READONLY":
        return UC_PROT_READ
    else:
        return UC_PROT_ALL

def load_binary():
    dp = Dumpulator("devirtme1.vmp.dmp")
    mu = Uc(UC_ARCH_X86, UC_MODE_64)
    print("[*] loading dump into unicorn and triton dse engine")
    mappings = dp.memory.map()
    for m in mappings:
        prot = get_map_prot(m.protect.name)
        base = m.base
        size = m.region_size
        if prot == UC_PROT_NONE:
            continue
        if len(m.info) > 0:
            if m.info[0] == "PEB":
                print("[*](unicorn) loading peb into unicorn")
                mu.mem_map(0, size, prot)
                mu.mem_write(0,bytes(dp.memory.read(base,size)))
                continue
        print(f"[*](unicorn) mapping memory region: va_range={hex(base)}-{hex(base+size)}, perms: {m.protect.name}")
        mu.mem_map(base, size, prot)
        print(f"[*](unicorn) writing data into region: va_range={hex(base)}-{hex(base+size)}, perms: {m.protect.name}")
        data = bytes(dp.memory.read(base, size))
        mu.mem_write(base, data)
    print("[*](unicorn) initializing cpu registers")
    mu.reg_write(UC_X86_REG_RAX,dp.regs.rax)
    mu.reg_write(UC_X86_REG_RCX,dp.regs.rcx)
    mu.reg_write(UC_X86_REG_RDX,dp.regs.rdx)
    mu.reg_write(UC_X86_REG_RBX,dp.regs.rbx)
    mu.reg_write(UC_X86_REG_RSP,dp.regs.rsp)
    mu.reg_write(UC_X86_REG_RBP,dp.regs.rbp)
    mu.reg_write(UC_X86_REG_RSI,dp.regs.rsi)
    mu.reg_write(UC_X86_REG_RDI,dp.regs.rdi)
    mu.reg_write(UC_X86_REG_R8,dp.regs.r8)
    mu.reg_write(UC_X86_REG_R9,dp.regs.r9)
    mu.reg_write(UC_X86_REG_R10,dp.regs.r10)
    mu.reg_write(UC_X86_REG_R11,dp.regs.r11)
    mu.reg_write(UC_X86_REG_R12,dp.regs.r12)
    mu.reg_write(UC_X86_REG_R13,dp.regs.r13)
    mu.reg_write(UC_X86_REG_R14,dp.regs.r14)
    mu.reg_write(UC_X86_REG_R15,dp.regs.r15)
    print("[+] complete init execution context")
    return mu,dp.regs.rip

def init_triton_dse(mu):
    print("[*](triton dse) init ctx")
    ctx = TritonContext(ARCH.X86_64)
    ctx.setMode(MODE.ALIGNED_MEMORY, True)
    ctx.setMode(MODE.AST_OPTIMIZATIONS, True)
    ctx.setMode(MODE.CONSTANT_FOLDING, True)
    ctx.setConcreteRegisterValue(ctx.registers.rax,mu.reg_read(UC_X86_REG_RAX))
    ctx.setConcreteRegisterValue(ctx.registers.rcx,mu.reg_read(UC_X86_REG_RCX))
    ctx.setConcreteRegisterValue(ctx.registers.rdx,mu.reg_read(UC_X86_REG_RDX))
    ctx.setConcreteRegisterValue(ctx.registers.rbx,mu.reg_read(UC_X86_REG_RBX))
    ctx.setConcreteRegisterValue(ctx.registers.rsp,mu.reg_read(UC_X86_REG_RSP))
    ctx.setConcreteRegisterValue(ctx.registers.rbp,mu.reg_read(UC_X86_REG_RBP))
    ctx.setConcreteRegisterValue(ctx.registers.rsi,mu.reg_read(UC_X86_REG_RSI))
    ctx.setConcreteRegisterValue(ctx.registers.rdi,mu.reg_read(UC_X86_REG_RDI))
    ctx.setConcreteRegisterValue(ctx.registers.r8,mu.reg_read(UC_X86_REG_R8))
    ctx.setConcreteRegisterValue(ctx.registers.r9,mu.reg_read(UC_X86_REG_R9))
    ctx.setConcreteRegisterValue(ctx.registers.r10,mu.reg_read(UC_X86_REG_R10))
    ctx.setConcreteRegisterValue(ctx.registers.r11,mu.reg_read(UC_X86_REG_R11))
    ctx.setConcreteRegisterValue(ctx.registers.r12,mu.reg_read(UC_X86_REG_R12))
    ctx.setConcreteRegisterValue(ctx.registers.r13,mu.reg_read(UC_X86_REG_R13))
    ctx.setConcreteRegisterValue(ctx.registers.r14,mu.reg_read(UC_X86_REG_R14))
    ctx.setConcreteRegisterValue(ctx.registers.r15,mu.reg_read(UC_X86_REG_R15))
    for region in mu.mem_regions():
        print(f"[*](triton dse) mapping and writing data into region: va_range={hex(region[0])}-{hex(region[1])}")
        data = mu.mem_read(region[0],region[1]-region[0])
        ctx.setConcreteMemoryAreaValue(region[0], data)
    return ctx

def deobfuscate_reg_trace(ctx, reg, handle_address, control_flow_insn):
    print(f"[*](triton dse) backward slicing {reg.getName()} reg")
    sliced_reg = ctx.sliceExpressions(ctx.getRegisterAst(reg).getSymbolicExpression())
    keys = sorted(sliced_reg)
    disasm_line = ""
    for key in keys:
        line = sliced_reg[key].getDisassembly()
        if len(line) == 0:
            continue
        disasm_line+=line+"\n"
    disasm_line+=f"{hex(control_flow_insn.getAddress())}: {control_flow_insn.getDisassembly()}\n"
    print(f"[*](triton dse) synthesize {reg.getName()} sliced ast..")
    synth = ctx.synthesize(sliced_reg[keys[-1]].getAst(),constant=True,subexpr=True)
    print("[*](triton dse) lifting to LLVM..")
    if synth:
        llvm_ir = ctx.liftToLLVM(synth, fname=f"hdl_{hex(handle_address)[2:]}",optimize=True)
    else:
        llvm_ir = ctx.liftToLLVM(sliced_reg[keys[-1]].getAst(), fname=f"hdl_{hex(handle_address)[2:]}",optimize=True)
    for symvar in list(ctx.getSymbolicVariables().values()):
        if llvm_ir.find(symvar.getName()) != -1:
            llvm_ir = llvm_ir.replace(symvar.getName(),symvar.getAlias())
    return disasm_line, llvm_ir

def symbolizeMemIfContainsBaseReg(mu,ctx,insn,reg,reg_id):
    for op in insn.getOperands():
            if op.getType() == OPERAND.MEM:
                if op.getBaseRegister() == reg:
                    address = mu.reg_read(reg_id)
                    mem_access = MemoryAccess(address, op.getSize())
                    print("[*] symbolizing mem",mem_access)
                    ctx.symbolizeMemory(mem_access,f"mem_{reg.getName()}_{hex(address)[2:]}")

handle_address = 0
pop_regs = []
pco_reg = None
def hook_insn(mu, address, size, ctx):
    global pco_reg
    global pop_regs
    global handle_address
    if len(pop_regs) == 16:
        print("[+] found vmexit\n[+] done emulation")
        mu.emu_stop()
        return
    if handle_address == 0:
        handle_address = address
    insn = Instruction(address,bytes(mu.mem_read(address,size+1)))
    ctx.processing(insn)
    print(f"{hex(insn.getAddress())}: {insn.getDisassembly()}")
    opcode = insn.getType()
    if opcode == OPCODE.X86.RET:
        if not pco_reg:
            print("[+] found not explorated branch\n[+] done emulation")    
            mu.emu_stop()
            return
        disasm_line,llvm_ir = deobfuscate_reg_trace(ctx,pco_reg,handle_address,insn)
        print("[*] check if exists..")        
        if os.path.exists(f"hdl_{hex(handle_address)[2:]}/"):
            print("[-] found element, skipping")
        else:
            print("[+] saving to handles array")
            os.mkdir(f"hdl_{hex(handle_address)[2:]}/")
            with open(f"hdl_{hex(handle_address)[2:]}/hdl_{hex(handle_address)[2:]}.asm",'w') as f: f.write(disasm_line)
            with open(f"hdl_{hex(handle_address)[2:]}/hdl_{hex(handle_address)[2:]}.ll",'w') as f: f.write(llvm_ir)
        ctx.reset()
        pop_regs.clear()
        handle_address = 0
        pco_reg = None
        ctx = init_triton_dse(mu)
        return
    else:
        pco_reg = None
    try:
        op1 = insn.getOperands()[0]
    except:
        return
    
    if opcode == OPCODE.X86.JMP and op1.getType() == OPERAND.REG:
        disasm_line,llvm_ir = deobfuscate_reg_trace(ctx,op1,handle_address,insn)
        print("[*] check if exists..")        
        if os.path.exists(f"hdl_{hex(handle_address)[2:]}/"):
            print("[-] found element, skipping")
        else:
            print("[+] saving to handles array")
            os.mkdir(f"hdl_{hex(handle_address)[2:]}/")
            with open(f"hdl_{hex(handle_address)[2:]}/hdl_{hex(handle_address)[2:]}.asm",'w') as f: f.write(disasm_line)
            with open(f"hdl_{hex(handle_address)[2:]}/hdl_{hex(handle_address)[2:]}.ll",'w') as f: f.write(llvm_ir)
        ctx.reset()
        pop_regs.clear()
        handle_address = 0
        ctx = init_triton_dse(mu)
    elif opcode == OPCODE.X86.PUSH and op1.getType() == OPERAND.REG:
        pco_reg = op1
        for reg in pop_regs:
            if reg == op1.getName():
                pop_regs.remove(reg)
    elif opcode == OPCODE.X86.POP and op1.getType()  == OPERAND.REG:
        print("[+] found ",insn.getDisassembly())
        for reg in pop_regs:
            if reg == op1.getName():
                return
        pop_regs.append(op1.getName())


def hook_mem_access(mu, access, address, size, value, ctx):
    if access == UC_MEM_WRITE:
        ctx.setConcreteMemoryAreaValue(address, bytes(mu.mem_read(address,size)))

   
def symbolize_mem_hook_insn(mu, address, size, ctx):
    global pco_reg
    global pop_regs
    if len(pop_regs) == 16:
        print("[+] found vmexit\n[+] done emulation")
        mu.emu_stop()
        return
    insn = Instruction(address,bytes(mu.mem_read(address,size+1)))
    ctx.disassembly(insn)
    opcode = insn.getType()
    if opcode == OPCODE.X86.LEA:
        return
    symbolizeMemIfContainsBaseReg(mu, ctx, insn, ctx.registers.rsp, UC_X86_REG_RSP)
    symbolizeMemIfContainsBaseReg(mu, ctx, insn, ctx.registers.rbx, UC_X86_REG_RBX)
    symbolizeMemIfContainsBaseReg(mu, ctx, insn, ctx.registers.r10, UC_X86_REG_R10)
    if opcode == OPCODE.X86.RET:
        if not pco_reg:
            print("[+] found not explorated branch\n[+] done emulation")    
            mu.emu_stop()
            return
        pop_regs.clear()
        pco_reg = None
        return
    else:
        pco_reg = None
    try:
        op1 = insn.getOperands()[0]
    except:
        return
    if opcode == OPCODE.X86.PUSH and op1.getType() == OPERAND.REG:
        pco_reg = op1
        for reg in pop_regs:
            if reg == op1.getName():
                pop_regs.remove(reg)
    elif opcode == OPCODE.X86.POP and op1.getType()  == OPERAND.REG:
        print("[+] found ",insn.getDisassembly())
        for reg in pop_regs:
            if reg == op1.getName():
                return
        pop_regs.append(op1.getName())


sym_vars_mu, rip = load_binary()
devirt_mu, rip = load_binary()

ctx = init_triton_dse(sym_vars_mu)

sym_vars_mu.hook_add(UC_HOOK_CODE, symbolize_mem_hook_insn, ctx)
sym_vars_mu.emu_start(begin=rip, until=-1)

devirt_mu.hook_add(UC_HOOK_CODE, hook_insn, ctx)
devirt_mu.hook_add(UC_HOOK_MEM_WRITE, hook_mem_access, ctx)
devirt_mu.emu_start(begin=rip, until=-1)