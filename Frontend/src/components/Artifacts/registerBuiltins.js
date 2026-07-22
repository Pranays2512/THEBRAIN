import{

registerArtifact

}

from

"../../lib/artifacts/ArtifactRegistry";

import CodeArtifact

from "./CodeArtifact";

import MarkdownArtifact

from "./MarkdownArtifact";

import ImageArtifact

from "./ImageArtifact";

registerArtifact(

"code",

CodeArtifact

);

registerArtifact(

"markdown",

MarkdownArtifact

);

registerArtifact(

"image",

ImageArtifact

);
