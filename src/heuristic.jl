
mutable struct Heuristic

    alloc::Bool
    ptr::Ptr{Void}
    net::Network

    function Function(a, ptr, n)
        this = new(a, ptr, n)
        finalizer(this, dealloc)
        this
    end
end

function Heuristic(name::String, net::Network)
    
    if name == "PVPQ switching"
        Function(true, ccall((:HEUR_PVPQ_new, libpfnet), Ptr{Void}, (Ptr{Void},), net.ptr), net)
    else
        throw(ArgumentError("invalid heuristic name"))
    end
end

function dealloc(heur::Heuristic)
    if heur.alloc
        ccall((:HEUR_del, libpfnet), Void, (Ptr{Void},), heur.ptr)
    end
    heur.alloc = false
    heur.ptr = C_NULL
end
