import { useContext } from "react";
import { MaterialContext } from "./MaterialContext";

export default function useMaterial() {

    return useContext(MaterialContext);

}
