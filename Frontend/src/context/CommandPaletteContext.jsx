import {
    createContext,
    useContext,
    useMemo,
    useState
} from "react";

const CommandPaletteContext=createContext();

export function CommandPaletteProvider({

    children

}){

    const[open,setOpen]=useState(false);

    const value=useMemo(()=>({

        open,

        openPalette(){

            setOpen(true);

        },

        closePalette(){

            setOpen(false);

        },

        togglePalette(){

            setOpen(v=>!v);

        }

    }),[open]);

    return(

        <CommandPaletteContext.Provider

            value={value}

        >

            {children}

        </CommandPaletteContext.Provider>

    );

}

export function useCommandPalette(){

    return useContext(

        CommandPaletteContext

    );

}
